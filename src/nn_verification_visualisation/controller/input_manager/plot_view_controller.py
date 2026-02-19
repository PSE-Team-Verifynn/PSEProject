from __future__ import annotations

from logging import Logger
import threading
from time import sleep
from typing import TYPE_CHECKING

from PySide6.QtCore import QTimer, SignalInstance, QMetaObject, Qt
from PySide6.QtWidgets import QApplication
from onnx import ModelProto

import numpy as np

from multiprocessing import Process, Queue

from nn_verification_visualisation.controller.process_manager.algorithm_executor import AlgorithmExecutor
from nn_verification_visualisation.model.data_loader.algorithm_file_observer import AlgorithmFileObserver
from nn_verification_visualisation.model.data.diagram_config import DiagramConfig
from nn_verification_visualisation.model.data.plot_generation_config import PlotGenerationConfig
from nn_verification_visualisation.model.data.storage import Storage
from nn_verification_visualisation.utils.result import Result, Failure, Success
from nn_verification_visualisation.view.dialogs.plot_config_dialog import PlotConfigDialog
from nn_verification_visualisation.view.plot_view.comparison_loading_widget import ComparisonLoadingWidget
from nn_verification_visualisation.view.plot_view.plot_page import PlotPage
from nn_verification_visualisation.view.plot_view.status import Status

if TYPE_CHECKING:
    from nn_verification_visualisation.view.plot_view.plot_view import PlotView


def execute_algorithm_wrapper(index, queue, model: ModelProto, input_bounds: np.ndarray, algorithm_path: str,
                              selected_neurons: list[tuple[int, int]]) -> None:
    try:
        executor = AlgorithmExecutor()
        execution_res = executor.execute_algorithm(model, input_bounds, algorithm_path,
                                                   selected_neurons)

        if not execution_res.is_success:
            queue.put((index, Failure(execution_res.error)))
            return

        output_bound_np, directions = execution_res.data

        if output_bound_np.shape[1] != 2:
            queue.put((index, Failure(Exception(f"Algorithm returned false bounds"))))
            return

        output_bounds = []
        for bounds in output_bound_np.tolist():
            output_bounds.append((bounds[0], bounds[1]))

        # Send back tuple: (index, Result)
        queue.put((index, Success((output_bounds, directions))))

    except Exception as e:
        queue.put((index, Failure(e)))


class PlotViewController:
    """
    Class representing a plot view.
    """
    logger = Logger(__name__)
    current_plot_view: PlotView
    card_size: int
    plot_titles: list[str]
    diagram_selections: dict[str, set[int]]

    def __init__(self, current_plot_view: PlotView):
        self.current_plot_view = current_plot_view
        self.card_size = 420
        self.plot_titles = []
        self.node_pairs = []
        self.node_pair_bounds = []
        self.diagram_selections = {}

        # start listening for algorithm changes
        AlgorithmFileObserver()

    def change_plot(self, plot_index: int | str, add: bool, pair_index: int):
        title: str | None
        if isinstance(plot_index, int):
            if plot_index < 0 or plot_index >= len(self.plot_titles):
                return
            title = self.plot_titles[plot_index]
        else:
            title = plot_index
        if title is None:
            return
        selection = self.diagram_selections.setdefault(title, set())
        if add:
            selection.add(pair_index)
        else:
            selection.discard(pair_index)

    def start_computation(self, plot_generation_configs: list[PlotGenerationConfig]):
        """
        Starts the main calculation.
        :param plot_generation_configs: plot configs
        """
        logger = Logger(__name__)

        polygons: list[list[tuple[float, float]] | None] = [None] * len(plot_generation_configs)

        result_queue = Queue()
        algorithm_processes: list[Process | None] = []

        diagram_config = DiagramConfig(plot_generation_configs, polygons)

        def terminate_algorithm_process(process_index: int) -> bool:
            if process_index >= len(algorithm_processes) or not algorithm_processes[process_index]:
                return False

            process = algorithm_processes[process_index]

            if process.is_alive():
                logger.info(f"Terminating algorithm process {process_index}")
                process.terminate()
                process.join()
                result_queue.put((process_index, Failure(Exception("Cancelled by User"))))
                return True

            return False

        def result_listener():
            results_received = 0
            total_tasks = len(plot_generation_configs)

            while results_received < total_tasks:
                # wait for a result from the queue
                result_index, result = result_queue.get()

                print(f"RESULT: {result_index}: {result.is_success}")

                if result.is_success:
                    bounds_list, directions_list = result.data

                    polygons[result_index] = self.compute_polygon(bounds_list, directions_list)
                else:
                    logger.error(f"Algorithm {result_index} failed: {result.error}")

                results_received += 1
                loading_screen.on_update.emit((result_index, result))

            loading_screen.loading_finished()
            logger.info("All computations finished/cancelled.")

            print(f"Done: {results_received}/{total_tasks}, \n Polygons {str(polygons)}")

        # start algorithm processes
        for index, plot_generation_config in enumerate(plot_generation_configs):
            model: ModelProto = plot_generation_config.nnconfig.network.model
            input_bounds: np.ndarray = AlgorithmExecutor.input_bounds_to_numpy(plot_generation_config.nnconfig.bounds)
            algorithm_path: str = plot_generation_config.algorithm.path
            selected_neurons: list[tuple[int, int]] = plot_generation_config.selected_neurons

            new_process = Process(target=execute_algorithm_wrapper,
                                  args=(index, result_queue, model, input_bounds, algorithm_path, selected_neurons), )
            algorithm_processes.append(new_process)
            new_process.start()

        loading_screen = ComparisonLoadingWidget(diagram_config, self, terminate_algorithm_process)

        listener = threading.Thread(target=result_listener)
        listener.daemon = True
        listener.start()

        self.current_plot_view.add_loading_tab(loading_screen)

    def create_diagram_tab(self, base: ComparisonLoadingWidget):
        # remove failed algorithms from diagram config:
        diagram_config = base.diagram_config
        for i in range(len(diagram_config.polygons) - 1, -1, -1):
            if diagram_config.polygons[i] is None:
                diagram_config.polygons.pop(i)
                diagram_config.plot_generation_configs.pop(i)

        # add to storage
        storage = Storage()
        storage.diagrams.append(diagram_config)
        storage.request_autosave()

        # replace ComparisonLoadingWidget-Tab with PlotPage-Tab
        tab_index = self.current_plot_view.tabs.indexOf(base)
        self.current_plot_view.tabs.close_tab(tab_index)
        self.current_plot_view.tabs.add_tab(PlotPage(self, diagram_config), index=tab_index)

    def open_plot_generation_dialog(self):
        dialog = PlotConfigDialog(self)
        self.current_plot_view.open_dialog(dialog)

    def open_plot_generation_editing_dialog(self, configs: list[PlotGenerationConfig], plot_page):
        close_callback = lambda: self.current_plot_view.close_tab(self.current_plot_view.tabs.indexOf(plot_page))
        dialog = PlotConfigDialog(self, (configs, close_callback))
        self.current_plot_view.open_dialog(dialog)

    def set_card_size(self, value: int):
        self.card_size = value

    def compute_polygon(
            self,
            bounds: list[tuple[float, float]],
            directions: list[tuple[float, ...]],
    ) -> list[tuple[float, float]] | list[list[tuple[float, float, float]]]:
        """
        Computes the polytope defined by the half-space constraints derived from
        directions and bounds.

        For 2 neurons (2D directions) returns a flat polygon:
            list[tuple[float, float]]

        For 3 neurons (3D directions) returns a list of triangular faces,
        each face being a list of three 3-D vertices:
            list[list[tuple[float, float, float]]]

        :param bounds:      list of (low, high) scalar bounds, one per direction
        :param directions:  list of direction vectors (length == number of neurons)
        :return:            polygon (2D) or triangulated polyhedron faces (3D)
        """
        if not bounds or not directions:
            return []

        dim = len(directions[0])

        # ------------------------------------------------------------------ 2D --
        if dim == 2:
            def clip_polygon(poly: list[tuple[float, float]], a: float, b: float, c: float):
                def inside(p: tuple[float, float]) -> bool:
                    return a * p[0] + b * p[1] <= c + 1e-9

                def intersect(p1: tuple[float, float], p2: tuple[float, float]):
                    x1, y1 = p1
                    x2, y2 = p2
                    dx, dy = x2 - x1, y2 - y1
                    denom = a * dx + b * dy
                    if abs(denom) < 1e-12:
                        return p2
                    t = (c - a * x1 - b * y1) / denom
                    return (x1 + t * dx, y1 + t * dy)

                out: list[tuple[float, float]] = []
                for i in range(len(poly)):
                    curr, prev = poly[i], poly[i - 1]
                    curr_in, prev_in = inside(curr), inside(prev)
                    if curr_in:
                        if not prev_in:
                            out.append(intersect(prev, curr))
                        out.append(curr)
                    elif prev_in:
                        out.append(intersect(prev, curr))
                return out

            max_bound = max(abs(v) for (low, high) in bounds for v in (low, high))
            m = max(5.0, max_bound * 2.0 + 1.0)
            poly: list[tuple[float, float]] = [(-m, -m), (m, -m), (m, m), (-m, m)]
            for i, (low, high) in enumerate(bounds):
                a, b = directions[i]
                poly = clip_polygon(poly, a, b, high)
                if not poly:
                    break
                poly = clip_polygon(poly, -a, -b, -low)
                if not poly:
                    break
            return poly

        # ------------------------------------------------------------------ 3D --
        if dim == 3:

            def find_interior_point(
                    hs_normals: np.ndarray, hs_offsets: np.ndarray
            ) -> np.ndarray | None:
                """
                Find a point strictly inside all half-spaces  n·x <= d.
                Uses a least-squares seed then iteratively pushes away from
                violated constraints.  Returns None if the system is infeasible.
                """
                try:
                    x, _, _, _ = np.linalg.lstsq(hs_normals, hs_offsets, rcond=None)
                except Exception:
                    x = np.zeros(3)

                for _ in range(300):
                    violations = hs_normals @ x - hs_offsets  # positive => violated
                    if np.all(violations <= -1e-8):
                        return x
                    worst = int(np.argmax(violations))
                    x = x - (violations[worst] + 0.1) * hs_normals[worst]

                violations = hs_normals @ x - hs_offsets
                return x if np.all(violations <= 1e-6) else None

            def triangulate_convex_hull(
                    pts: np.ndarray,
            ) -> list[list[int]]:
                """
                Incremental convex hull (Beneath-Beyond algorithm).
                Returns triangular faces as lists of vertex indices,
                with outward-facing winding order.
                pts: np.ndarray shape (N, 3)
                """

                def make_face(
                        i: int, j: int, k: int, interior: np.ndarray
                ) -> tuple[list[int], np.ndarray, float] | None:
                    a, b, c = pts[i], pts[j], pts[k]
                    normal = np.cross(b - a, c - a).astype(float)
                    nn = np.linalg.norm(normal)
                    if nn < 1e-12:
                        return None
                    normal /= nn
                    offset = float(normal @ a)
                    if normal @ interior > offset:
                        # flip so interior is on the n·x < offset side
                        normal, offset = -normal, -offset
                        return ([i, k, j], normal, offset)
                    return ([i, j, k], normal, offset)

                def find_initial_tetra() -> tuple[int, int, int, int] | None:
                    p0 = pts[0]
                    dists = np.linalg.norm(pts - p0, axis=1)
                    i1 = int(np.argmax(dists))
                    if dists[i1] < 1e-10:
                        return None
                    crosses = np.cross(pts - p0, pts[i1] - p0)
                    i2 = int(np.argmax(np.linalg.norm(crosses, axis=1)))
                    if np.linalg.norm(crosses[i2]) < 1e-10 or i2 == i1:
                        return None
                    tri_normal = np.cross(pts[i1] - p0, pts[i2] - p0).astype(float)
                    tri_normal /= np.linalg.norm(tri_normal)
                    dists_plane = np.abs((pts - p0) @ tri_normal)
                    i3 = int(np.argmax(dists_plane))
                    if dists_plane[i3] < 1e-10 or i3 in (0, i1, i2):
                        return None
                    return (0, i1, i2, i3)

                n = len(pts)
                if n < 4:
                    return []

                tetra = find_initial_tetra()
                if tetra is None:
                    return []

                i0, i1, i2, i3 = tetra
                interior = pts[[i0, i1, i2, i3]].mean(axis=0)

                faces: list[tuple[list[int], np.ndarray, float]] = []
                for (a, b, c) in [(i0, i1, i2), (i0, i1, i3), (i0, i2, i3), (i1, i2, i3)]:
                    f = make_face(a, b, c, interior)
                    if f is not None:
                        faces.append(f)

                processed = {i0, i1, i2, i3}

                for idx in range(n):
                    if idx in processed:
                        continue
                    p = pts[idx]

                    # Faces visible from the new point
                    visible = [f for f in faces if f[1] @ p > f[2] + 1e-9]
                    if not visible:
                        processed.add(idx)
                        continue

                    # Horizon: edges shared by exactly one visible face
                    edge_count: dict[tuple[int, int], int] = {}
                    for verts, _, _ in visible:
                        for k in range(3):
                            edge = tuple(sorted((verts[k], verts[(k + 1) % 3])))
                            edge_count[edge] = edge_count.get(edge, 0) + 1
                    horizon = [e for e, cnt in edge_count.items() if cnt == 1]

                    faces = [f for f in faces if f not in visible]
                    for (a, b) in horizon:
                        f = make_face(a, b, idx, interior)
                        if f is not None:
                            faces.append(f)

                    processed.add(idx)

                return [verts for (verts, _, _) in faces]

            # Build half-space system:  n·x <= d
            hs_normals_list: list[np.ndarray] = []
            hs_offsets_list: list[float] = []
            for (low, high), d in zip(bounds, directions):
                hs_normals_list.append(np.array(d, dtype=float))
                hs_offsets_list.append(float(high))
                hs_normals_list.append(-np.array(d, dtype=float))
                hs_offsets_list.append(float(-low))

            hs_normals = np.array(hs_normals_list)
            hs_offsets = np.array(hs_offsets_list)

            interior = find_interior_point(hs_normals, hs_offsets)
            if interior is None:
                return []  # infeasible / empty polytope

            # Vertices: intersect every triple of bounding planes,
            # keep only those satisfying all other half-spaces.
            n_hs = len(hs_normals)
            candidate_vertices: list[np.ndarray] = []
            for i in range(n_hs):
                for j in range(i + 1, n_hs):
                    for k in range(j + 1, n_hs):
                        A = np.array([hs_normals[i], hs_normals[j], hs_normals[k]])
                        b_vec = np.array([hs_offsets[i], hs_offsets[j], hs_offsets[k]])
                        if abs(np.linalg.det(A)) < 1e-10:
                            continue
                        try:
                            pt = np.linalg.solve(A, b_vec)
                        except np.linalg.LinAlgError:
                            continue
                        if np.all(hs_normals @ pt <= hs_offsets + 1e-8):
                            candidate_vertices.append(pt)

            if len(candidate_vertices) < 4:
                return []

            # Deduplicate vertices
            unique: list[np.ndarray] = [candidate_vertices[0]]
            for v in candidate_vertices[1:]:
                if all(np.linalg.norm(v - u) > 1e-8 for u in unique):
                    unique.append(v)
            if len(unique) < 4:
                return []

            pts_array = np.array(unique)
            face_indices = triangulate_convex_hull(pts_array)

            return [
                [tuple(pts_array[vi].tolist()) for vi in face]
                for face in face_indices
            ]

        raise ValueError(
            f"compute_polygon only supports 2D or 3D directions, got dim={dim}"
        )
