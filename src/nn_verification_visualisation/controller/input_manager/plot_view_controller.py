from __future__ import annotations

import itertools
from logging import Logger
import threading
from typing import TYPE_CHECKING

from onnx import ModelProto

import numpy as np

from multiprocessing import Process, Queue

from nn_verification_visualisation.controller.process_manager.algorithm_executor import AlgorithmExecutor
from nn_verification_visualisation.model.data_loader.algorithm_file_observer import AlgorithmFileObserver
from nn_verification_visualisation.model.data.diagram_config import DiagramConfig
from nn_verification_visualisation.model.data.plot_generation_config import PlotGenerationConfig
from nn_verification_visualisation.model.data.storage import Storage
from nn_verification_visualisation.utils.result import Failure, Success
from nn_verification_visualisation.view.dialogs.plot_config_dialog import PlotConfigDialog
from nn_verification_visualisation.view.plot_view.comparison_loading_widget import ComparisonLoadingWidget
from nn_verification_visualisation.view.plot_view.plot_page import PlotPage

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

        polygons: list[list[tuple[float, ...]] | None] = [None] * len(plot_generation_configs)

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

                    polygons[result_index] = self.compute_polytope_vertices(bounds_list, directions_list)
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
            nn_config = plot_generation_config.nnconfig
            bounds_model = nn_config.bounds
            if 0 <= plot_generation_config.bounds_index < len(nn_config.saved_bounds):
                bounds_model = nn_config.saved_bounds[plot_generation_config.bounds_index]
            input_bounds: np.ndarray = AlgorithmExecutor.input_bounds_to_numpy(bounds_model)
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

    def compute_polytope_vertices(
        self, bounds: list[tuple[float, float]], directions: list[tuple[float, ...]]) -> list[tuple[float, ...]]:
        """
        Computes vertices of a convex polytope from directional lower/upper bounds.
        """
        if not bounds or not directions:
            return []

        dimension = len(directions[0])
        if dimension <= 0:
            return []

        eps = 1e-8
        inequalities: list[tuple[np.ndarray, float]] = []
        for low_high, direction in zip(bounds, directions):
            low, high = low_high
            d = np.asarray(direction, dtype=float)
            inequalities.append((d, float(high)))
            inequalities.append((-d, float(-low)))

        max_bound = max(abs(v) for (low, high) in bounds for v in (low, high))
        outer = max(5.0, max_bound * 2.0 + 1.0)
        for axis in range(dimension):
            unit = np.zeros(dimension, dtype=float)
            unit[axis] = 1.0
            inequalities.append((unit, outer))
            inequalities.append((-unit, outer))

        vertices: list[np.ndarray] = []
        for combo in itertools.combinations(range(len(inequalities)), dimension):
            a_eq = np.stack([inequalities[idx][0] for idx in combo], axis=0)
            c_eq = np.asarray([inequalities[idx][1] for idx in combo], dtype=float)

            if np.linalg.matrix_rank(a_eq) < dimension:
                continue
            try:
                x = np.linalg.solve(a_eq, c_eq)
            except np.linalg.LinAlgError:
                continue

            if all(float(np.dot(a, x)) <= c + eps for a, c in inequalities):
                vertices.append(x)

        if not vertices:
            return []

        unique: list[tuple[float, ...]] = []
        seen = set()
        for v in vertices:
            key = tuple(np.round(v, 7))
            if key in seen:
                continue
            seen.add(key)
            unique.append(tuple(float(x) for x in v))

        return unique
