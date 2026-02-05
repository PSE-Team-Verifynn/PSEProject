from __future__ import annotations

import logging
from logging import Logger
from typing import TYPE_CHECKING

import numpy as np

from nn_verification_visualisation.controller.process_manager.algorithm_executor import AlgorithmExecutor
from nn_verification_visualisation.model.data_loader.algorithm_file_observer import AlgorithmFileObserver
from nn_verification_visualisation.model.data.diagram_config import DiagramConfig
from nn_verification_visualisation.model.data.plot_generation_config import PlotGenerationConfig
from nn_verification_visualisation.model.data.storage import Storage
from nn_verification_visualisation.view.dialogs.plot_config_dialog import PlotConfigDialog

if TYPE_CHECKING:
    from nn_verification_visualisation.view.plot_view.plot_view import PlotView

class PlotViewController:
    logger = Logger(__name__)
    current_plot_view: PlotView
    current_tab: int
    card_size: int
    plot_titles: list[str]
    diagram_selections: dict[str, set[int]]

    def __init__(self, current_plot_view: PlotView):
        self.current_plot_view = current_plot_view
        self.current_tab = 0
        self.card_size = 420
        self.plot_titles = []
        self.node_pairs = []
        self.node_pair_bounds = []
        self.node_pair_colors = []
        self.diagram_selections = {}

        #start listening for algorithm changes
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
        polygons = []
        for plot_generation_config in plot_generation_configs:
            execution_res = AlgorithmExecutor.execute_algorithm(AlgorithmExecutor(), plot_generation_config)
            if not execution_res.is_success:
                logger.error(f"Could not execute algorithm: {execution_res.error}")
                print(f"Could not execute algorithm: {execution_res.error}")
                continue
            output_bound_np, directions = execution_res.data
            if output_bound_np.shape[1] != 2:
                print(f"Algorithm returned false bounds: {output_bound_np}")
            output_bounds = []
            for bounds in output_bound_np.tolist():
                output_bounds.append((bounds[0], bounds[1]))
            print(f"Got bounds: {output_bounds}")
            polygons.append(self.compute_polygon(output_bounds, directions))
            print(f"Computed polygon: {polygons[-1]}")

        diagram_config = DiagramConfig(plot_generation_configs,polygons)
        print("Generated Diagram Config")
        storage = Storage()
        storage.diagrams.append(diagram_config)
        self.current_plot_view.add_plot_tab(diagram_config)

    def change_tab(self, index: int):
        pass

    def open_plot_generation_dialog(self):
        dialog = PlotConfigDialog(self)
        self.current_plot_view.open_dialog(dialog)

    def set_card_size(self, value: int):
        self.card_size = value

    def register_plot(self, title: str):
        if title in self.plot_titles:
            return
        self.plot_titles.append(title)
        self.diagram_selections.setdefault(title, set())

    def remove_plot(self, title: str):
        if title in self.plot_titles:
            self.plot_titles.remove(title)
        self.diagram_selections.pop(title, None)
    def get_node_pairs(self) -> list[str]:
        return list(self.node_pairs)

    def get_node_pair_bounds(self, index: int) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        return self.node_pair_bounds[index]

    def get_node_pair_colors(self, index: int) -> tuple[str, str]:
        return self.node_pair_colors[index]

    def get_selection(self, title: str) -> set[int]:
        return set(self.diagram_selections.get(title, set()))


    def compute_polygon(
        self, bounds: list[tuple[float, float]], directions: list[tuple[float, float]]) -> list[tuple[float, float]]:
        def clip_polygon(poly: list[tuple[float, float]], a: float, b: float, c: float):
            def inside(p: tuple[float, float]) -> bool:
                return a * p[0] + b * p[1] <= c + 1e-9

            def intersect(p1: tuple[float, float], p2: tuple[float, float]):
                x1, y1 = p1
                x2, y2 = p2
                dx = x2 - x1
                dy = y2 - y1
                denom = a * dx + b * dy
                if abs(denom) < 1e-12:
                    return p2
                t = (c - a * x1 - b * y1) / denom
                return (x1 + t * dx, y1 + t * dy)

            out: list[tuple[float, float]] = []
            for i in range(len(poly)):
                curr = poly[i]
                prev = poly[i - 1]
                curr_in = inside(curr)
                prev_in = inside(prev)
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
            a,b = directions[i]
            poly = clip_polygon(poly, a, b, high)
            if not poly:
                break
            poly = clip_polygon(poly, -a, -b, -low)
            if not poly:
                break
        return poly
