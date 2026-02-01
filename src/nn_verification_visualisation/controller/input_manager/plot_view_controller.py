from __future__ import annotations

from typing import TYPE_CHECKING

from nn_verification_visualisation.model.data.algorithm_file_observer import AlgorithmFileObserver
from nn_verification_visualisation.model.data.diagram_config import DiagramConfig
from nn_verification_visualisation.model.data.plot import Plot
from nn_verification_visualisation.view.dialogs.plot_config_dialog import PlotConfigDialog

if TYPE_CHECKING:
    from nn_verification_visualisation.view.plot_view.plot_view import PlotView

class PlotViewController:
    current_plot_view: PlotView
    current_tab: int
    card_size: int
    plot_titles: list[str]
    node_pairs: list[str]
    node_pair_bounds: list[list[tuple[tuple[float, float], tuple[float, float]]]]
    node_pair_colors: list[tuple[str, str]]
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

    def start_computation(self, config: DiagramConfig):
        pass

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

    def add_node_pair(self, bounds: list[tuple[tuple[float, float], tuple[float, float]]]) -> int:
        pair_index = len(self.node_pairs)
        self.node_pairs.append(f"Node Pair {pair_index + 1}")
        self.node_pair_bounds.append(bounds)
        palette = [
            ("#59aef2", "#3b6ea8"),
            ("#7cc38d", "#3d7b57"),
            ("#f0b76f", "#a36b28"),
            ("#c08fd6", "#6e4d8c"),
            ("#f28fa2", "#9d3f50"),
            ("#7bd1d1", "#3a7a7a"),
        ]
        self.node_pair_colors.append(palette[pair_index % len(palette)])
        return pair_index

    def remove_node_pair(self, index: int):
        if index < 0 or index >= len(self.node_pair_bounds):
            return
        del self.node_pairs[index]
        del self.node_pair_bounds[index]
        if index < len(self.node_pair_colors):
            del self.node_pair_colors[index]
        for title, selection in self.diagram_selections.items():
            updated = set()
            for idx in selection:
                if idx == index:
                    continue
                if idx > index:
                    updated.add(idx - 1)
                else:
                    updated.add(idx)
            self.diagram_selections[title] = updated

    def get_node_pairs(self) -> list[str]:
        return list(self.node_pairs)

    def get_node_pair_bounds(self, index: int) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        return self.node_pair_bounds[index]

    def get_node_pair_colors(self, index: int) -> tuple[str, str]:
        return self.node_pair_colors[index]

    def get_selection(self, title: str) -> set[int]:
        return set(self.diagram_selections.get(title, set()))


    def compute_polygon(
        self, bounds: list[tuple[tuple[float, float], tuple[float, float]]]
    ) -> list[tuple[float, float]]:
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

        max_bound = max(abs(v) for _, (low, high) in bounds for v in (low, high))
        m = max(5.0, max_bound * 2.0 + 1.0)
        poly: list[tuple[float, float]] = [(-m, -m), (m, -m), (m, m), (-m, m)]

        for (a, b), (low, high) in bounds:
            poly = clip_polygon(poly, a, b, high)
            if not poly:
                break
            poly = clip_polygon(poly, -a, -b, -low)
            if not poly:
                break
        return poly
