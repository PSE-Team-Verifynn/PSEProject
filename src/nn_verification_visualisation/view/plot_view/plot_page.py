from typing import List

import math

from PySide6.QtCore import Qt, QEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from nn_verification_visualisation.model.data.diagram_config import DiagramConfig
from nn_verification_visualisation.view.plot_view.plot_widget import PlotWidget
from nn_verification_visualisation.view.base_view.tab import Tab

class PlotPage(Tab):
    configuration: DiagramConfig
    plots: List[PlotWidget]
    locked: List[PlotWidget]
    __plot_grid: QGridLayout
    __plot_map: dict[str, PlotWidget]
    __syncing: bool
    __card_size: int
    __scroll_area: QScrollArea | None
    __grid_host: QWidget | None
    __bottom_spacer_height: int
    __example_bounds: list[tuple[tuple[float, float], tuple[float, float]]]
    __primary_plot: PlotWidget | None
    __node_pairs: list[str]
    __node_pair_bounds: list[list[tuple[tuple[float, float], tuple[float, float]]]]
    __node_pair_colors: list[tuple[str, str]]
    __diagram_selections: dict[str, set[int]]
    __diagram_groups_layout: QVBoxLayout | None
    __node_pairs_list: QListWidget | None
    __node_pairs_layout: QVBoxLayout | None

    def __init__(self, configuration: DiagramConfig):
        self.__syncing = False
        self.__card_size = 420
        self.__scroll_area = None
        self.__grid_host = None
        self.__bottom_spacer_height = 32
        self.plots = []
        self.__plot_map = {}
        # Temporary example bounds: [((a, b), (lower, upper)), ...] for 4 directions.
        self.__example_bounds = [
            ((1.0, 0.0), (-2.6, 1.9)),
            ((0.0, 1.0), (-1.8, 3.4)),
            ((-1.0, 0.0), (-1.1, 2.4)),
            ((0.0, -1.0), (-2.2, 1.6)),
        ]
        self.__primary_plot = None
        # Temporary: store node pair bounds for sandboxed plotting.
        self.__node_pairs = []
        self.__node_pair_bounds = []
        self.__node_pair_colors = []
        self.__diagram_selections = {}
        self.__diagram_groups_layout = None
        self.__node_pairs_list = None
        self.__node_pairs_layout = None
        super().__init__("Example Tab")
        # configuration is currently not implemented
        # self.configuration = configuration

    def get_content(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        self.__scroll_area = scroll_area

        grid_host = QWidget()
        grid_host.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        self.__plot_grid = QGridLayout(grid_host)
        self.__plot_grid.setContentsMargins(4, 4, 4, 4)
        self.__plot_grid.setSpacing(12)
        self.__plot_grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.__grid_host = grid_host

        self.plots = []
        self.__plot_map = {}

        self.__register_node_pair(self.__example_bounds)
        plot_card = self.__create_plot_card("Diagram 01")
        self.plots.append(plot_card)
        self.__plot_map["Diagram 01"] = plot_card
        self.__primary_plot = plot_card
        self.__diagram_selections["Diagram 01"] = {0}
        self.__render_plot(plot_card)
        self.__rebuild_diagram_groups()

        scroll_area.setWidget(grid_host)
        layout.addWidget(scroll_area)

        scroll_area.viewport().installEventFilter(self)
        self.__relayout_plots()
        return container

    def get_side_bar(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        title = QLabel("Settings")
        title.setObjectName("title")
        layout.addWidget(title)

        self.__diagram_groups_layout = QVBoxLayout()
        self.__diagram_groups_layout.setContentsMargins(0, 0, 0, 0)
        self.__diagram_groups_layout.setSpacing(8)
        layout.addLayout(self.__diagram_groups_layout)
        self.__rebuild_diagram_groups()

        size_group = QGroupBox("Card Size")
        size_layout = QVBoxLayout(size_group)
        size_layout.setContentsMargins(6, 6, 6, 6)
        size_layout.setSpacing(4)
        # Temporary: will be replaced by persistent settings.
        size_slider = QSlider(Qt.Orientation.Horizontal)
        size_slider.setMinimum(320)
        size_slider.setMaximum(560)
        size_slider.setValue(self.__card_size)
        size_slider.setSingleStep(10)
        size_slider.valueChanged.connect(self.__on_card_size_changed)
        size_layout.addWidget(size_slider)
        layout.addWidget(size_group)

        node_pairs_group = QGroupBox("Node Pairs")
        node_pairs_layout = QVBoxLayout(node_pairs_group)
        node_pairs_layout.setContentsMargins(6, 6, 6, 6)
        node_pairs_layout.setSpacing(4)
        self.__node_pairs_layout = node_pairs_layout
        self.__node_pairs_list = QListWidget()
        node_pairs_layout.addWidget(self.__node_pairs_list)
        remove_pair_button = QPushButton("Remove Selected Pair")
        remove_pair_button.clicked.connect(self.__remove_selected_pair)
        node_pairs_layout.addWidget(remove_pair_button)
        self.__refresh_node_pairs_list()
        layout.addWidget(node_pairs_group)

        layout.addStretch(1)

        edit_button = QPushButton("Edit Comparison")
        layout.addWidget(edit_button, alignment=Qt.AlignmentFlag.AlignLeft)

        add_diagram_button = QPushButton("Add Diagram")
        add_diagram_button.clicked.connect(self.__add_diagram_from_current_bounds)
        layout.addWidget(add_diagram_button, alignment=Qt.AlignmentFlag.AlignLeft)

        temp_button = QPushButton("Bounds Playground")
        temp_button.clicked.connect(self.__open_temp_bounds_dialog)
        layout.addWidget(temp_button, alignment=Qt.AlignmentFlag.AlignLeft)
        # Temporary: bounds playground entry point (remove after validation).

        return container

    def change_lock(self, widget: PlotWidget):
        pass

    def fullscreen(self, widget: PlotWidget):
        if widget.canvas is None or widget.toolbar is None or widget.plot_layout is None:
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Fullscreen Plot")
        dialog_layout = QVBoxLayout(dialog)
        dialog_layout.setContentsMargins(6, 6, 6, 6)
        dialog_layout.setSpacing(6)

        widget.plot_layout.removeWidget(widget.toolbar)
        widget.plot_layout.removeWidget(widget.canvas)
        widget.toolbar.setParent(dialog)
        widget.canvas.setParent(dialog)
        dialog_layout.addWidget(widget.toolbar)
        dialog_layout.addWidget(widget.canvas, stretch=1)

        def restore():
            dialog_layout.removeWidget(widget.toolbar)
            dialog_layout.removeWidget(widget.canvas)
            widget.toolbar.setParent(widget)
            widget.canvas.setParent(widget)
            widget.plot_layout.addWidget(widget.toolbar)
            widget.plot_layout.addWidget(widget.canvas, stretch=1)

        dialog.finished.connect(restore)
        dialog.showMaximized()

    def export(self, widget: PlotWidget):
        pass

    def transform_changed(self, widget: PlotWidget, transform: tuple[float, float, float, float]):
        pass

    def __create_plot_card(
        self,
        title: str,
    ) -> PlotWidget:
        card = PlotWidget()
        card.plot_title = title
        card.setFixedSize(self.__card_size, self.__card_size)
        card.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(8, 8, 8, 8)
        card_layout.setSpacing(6)

        plot_placeholder = QFrame()
        plot_placeholder.setFrameShape(QFrame.Shape.StyledPanel)
        plot_placeholder.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        plot_layout = QVBoxLayout(plot_placeholder)
        plot_layout.setContentsMargins(4, 4, 4, 4)
        plot_layout.setSpacing(4)

        figure = Figure(figsize=(3.2, 2.4))
        canvas = FigureCanvas(figure)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        toolbar = NavigationToolbar(canvas, plot_placeholder)

        ax = figure.add_subplot(111)
        card.figure = figure
        card.axes = ax
        card.canvas = canvas
        card.toolbar = toolbar
        card.plot_layout = plot_layout
        card.locked = False
        card.polygon_patches = []
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.set_aspect("auto")

        plot_layout.addWidget(toolbar)
        plot_layout.addWidget(canvas, stretch=1)
        card_layout.addWidget(plot_placeholder)

        footer = QWidget()
        footer.setFixedHeight(40)
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(0, 0, 0, 0)
        footer_layout.setSpacing(6)

        footer_layout.addWidget(QLabel(title))
        footer_layout.addStretch(1)
        lock_button = QPushButton("Lock")
        lock_button.clicked.connect(lambda: self.__toggle_lock(card, lock_button))
        footer_layout.addWidget(lock_button)
        fullscreen_button = QPushButton("Fullscreen")
        fullscreen_button.clicked.connect(lambda: self.fullscreen(card))
        footer_layout.addWidget(fullscreen_button)

        ax.callbacks.connect("xlim_changed", lambda _ax: self.__on_limits_changed(card))
        ax.callbacks.connect("ylim_changed", lambda _ax: self.__on_limits_changed(card))

        card_layout.addWidget(footer)
        return card

    def __relayout_plots(self):
        if self.__scroll_area is None:
            return
        viewport_width = self.__scroll_area.viewport().width()
        margins = self.__plot_grid.contentsMargins()
        spacing = self.__plot_grid.horizontalSpacing()
        available_width = viewport_width - margins.left() - margins.right()
        if available_width <= 0:
            return
        if spacing < 0:
            spacing = 0
        columns = max(1, (available_width + spacing) // (self.__card_size + spacing))
        required_width = (
            columns * self.__card_size
            + max(0, columns - 1) * spacing
            + margins.left()
            + margins.right()
        )

        while self.__plot_grid.count():
            item = self.__plot_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(self.__grid_host)

        if self.__grid_host is not None:
            self.__grid_host.setFixedWidth(required_width)

        for index, widget in enumerate(self.plots):
            row = index // columns
            col = index % columns
            self.__plot_grid.addWidget(widget, row, col)

        spacer_row = (len(self.plots) + columns - 1) // columns
        spacer = QSpacerItem(0, self.__bottom_spacer_height, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.__plot_grid.addItem(spacer, spacer_row, 0, 1, columns)

    def __toggle_lock(self, widget: PlotWidget, button: QPushButton):
        widget.locked = not getattr(widget, "locked", False)
        button.setText("Unlock" if widget.locked else "Lock")

    def __on_card_size_changed(self, value: int):
        self.__card_size = value
        for widget in self.plots:
            widget.setFixedSize(self.__card_size, self.__card_size)
        self.__relayout_plots()

    def __register_node_pair(
        self, bounds: list[tuple[tuple[float, float], tuple[float, float]]]
    ) -> int:
        pair_index = len(self.__node_pairs)
        self.__node_pairs.append(f"Node Pair {pair_index + 1}")
        self.__node_pair_bounds.append(bounds)
        palette = [
            ("#59aef2", "#3b6ea8"),
            ("#7cc38d", "#3d7b57"),
            ("#f0b76f", "#a36b28"),
            ("#c08fd6", "#6e4d8c"),
            ("#f28fa2", "#9d3f50"),
            ("#7bd1d1", "#3a7a7a"),
        ]
        self.__node_pair_colors.append(palette[pair_index % len(palette)])
        self.__refresh_node_pairs_list()
        self.__rebuild_diagram_groups()
        return pair_index

    def __refresh_node_pairs_list(self):
        if self.__node_pairs_list is None:
            return
        self.__node_pairs_list.clear()
        self.__node_pairs_list.addItems(self.__node_pairs)

    def __remove_selected_pair(self):
        if self.__node_pairs_list is None:
            return
        row = self.__node_pairs_list.currentRow()
        if row < 0 or row >= len(self.__node_pair_bounds):
            return
        del self.__node_pairs[row]
        del self.__node_pair_bounds[row]
        if row < len(self.__node_pair_colors):
            del self.__node_pair_colors[row]
        for title, selection in self.__diagram_selections.items():
            updated = set()
            for idx in selection:
                if idx == row:
                    continue
                if idx > row:
                    updated.add(idx - 1)
                else:
                    updated.add(idx)
            self.__diagram_selections[title] = updated
        self.__refresh_node_pairs_list()
        self.__rebuild_diagram_groups()
        for plot in self.plots:
            self.__render_plot(plot)

    def __rebuild_diagram_groups(self):
        if self.__diagram_groups_layout is None:
            return
        while self.__diagram_groups_layout.count():
            item = self.__diagram_groups_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        for plot in self.plots:
            title = getattr(plot, "plot_title", "Diagram")
            group = QGroupBox(title)
            group.setObjectName("card")
            group_layout = QVBoxLayout(group)
            group_layout.setContentsMargins(6, 20, 6, 6)
            group_layout.setSpacing(0)
            selection = self.__diagram_selections.setdefault(title, set())
            for idx, pair_name in enumerate(self.__node_pairs):
                check_box = QCheckBox(pair_name)
                check_box.setChecked(idx in selection)
                check_box.stateChanged.connect(
                    lambda state, t=title, i=idx: self.__on_pair_toggled(t, i, state)
                )
                group_layout.addWidget(check_box)
            self.__diagram_groups_layout.addWidget(group)

    def __on_pair_toggled(self, plot_title: str, pair_index: int, state: int):
        selection = self.__diagram_selections.setdefault(plot_title, set())
        if state == Qt.CheckState.Checked.value:
            selection.add(pair_index)
        else:
            selection.discard(pair_index)
        plot = self.__plot_map.get(plot_title)
        if plot is not None:
            self.__render_plot(plot)

    def __render_plot(self, plot: PlotWidget):
        if plot.axes is None or plot.canvas is None:
            return
        ax = plot.axes
        ax.cla()
        ax.grid(True, alpha=0.2)
        title = getattr(plot, "plot_title", "Diagram")
        ax.set_title(title, fontsize=9)
        selection = self.__diagram_selections.get(title, set())
        all_points: list[tuple[float, float]] = []
        legend_handles = []
        legend_labels = []
        for order, pair_index in enumerate(sorted(selection)):
            if pair_index >= len(self.__node_pair_bounds):
                continue
            bounds = self.__node_pair_bounds[pair_index]
            poly_points = self.__compute_polygon(bounds)
            if not poly_points:
                continue
            all_points.extend(poly_points)
            face, edge = self.__node_pair_colors[pair_index]
            poly_array = np.array(poly_points)
            polygon = Polygon(poly_array, closed=True, facecolor=face, edgecolor=edge, alpha=0.6)
            ax.add_patch(polygon)
            legend_handles.append(polygon)
            legend_labels.append(self.__node_pairs[pair_index])
        if all_points:
            xs = [p[0] for p in all_points]
            ys = [p[1] for p in all_points]
            ax.set_xlim(min(xs) - 0.5, max(xs) + 0.5)
            ax.set_ylim(min(ys) - 0.5, max(ys) + 0.5)
        if legend_handles:
            ax.legend(legend_handles, legend_labels, loc="upper right", fontsize=7, frameon=True)
        plot.canvas.draw_idle()

    def __add_diagram_from_current_bounds(self):
        title = f"Diagram {len(self.plots) + 1:02d}"
        plot_card = self.__create_plot_card(title)
        self.plots.append(plot_card)
        self.__plot_map[title] = plot_card
        self.__diagram_selections[title] = set()
        self.__render_plot(plot_card)
        self.__rebuild_diagram_groups()
        self.__relayout_plots()

    def __compute_polygon(
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

    def __open_temp_bounds_dialog(self):
        if self.__primary_plot is None:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Temporary Bounds Playground")
        dialog_layout = QVBoxLayout(dialog)
        dialog_layout.setContentsMargins(12, 12, 12, 12)
        dialog_layout.setSpacing(10)

        limit_label = QLabel("Bound limit (abs)")
        limit_spin = QDoubleSpinBox()
        limit_spin.setDecimals(2)
        limit_spin.setRange(0.5, 10.0)
        limit_spin.setSingleStep(0.5)
        limit_spin.setValue(4.0)

        directions_label = QLabel("Directions (power of 2)")
        directions_value = QLabel()
        directions_slider = QSlider(Qt.Orientation.Horizontal)
        directions_slider.setMinimum(2)
        directions_slider.setMaximum(5)
        directions_slider.setSingleStep(1)
        directions_slider.setValue(3)

        def update_directions_label():
            directions_value.setText(f"{2 ** directions_slider.value()}")

        update_directions_label()
        directions_slider.valueChanged.connect(update_directions_label)

        bounds_view = QPlainTextEdit()
        bounds_view.setReadOnly(True)
        bounds_view.setMinimumHeight(120)

        latest_bounds: list[tuple[tuple[float, float], tuple[float, float]]] = []

        def generate_bounds():
            limit = limit_spin.value()
            direction_count = 2 ** directions_slider.value()
            directions = []
            for i in range(direction_count):
                angle = (2 * math.pi * i) / direction_count
                directions.append((math.cos(angle), math.sin(angle)))
            rng = np.random.default_rng()
            center_x = rng.uniform(-limit * 0.5, limit * 0.5)
            center_y = rng.uniform(-limit * 0.5, limit * 0.5)
            new_bounds: list[tuple[tuple[float, float], tuple[float, float]]] = []
            for a, b in directions:
                center_proj = a * center_x + b * center_y
                width = rng.uniform(limit * 0.2, limit * 0.6)
                low = center_proj - width * 0.5
                high = center_proj + width * 0.5
                low = max(-limit, low)
                high = min(limit, high)
                if low > high:
                    low, high = high, low
                if abs(high - low) < limit * 0.1:
                    high = min(limit, low + limit * 0.2)
                new_bounds.append(((a, b), (low, high)))
            return new_bounds

        def update_bounds_view(new_bounds: list[tuple[tuple[float, float], tuple[float, float]]]):
            bounds_lines = []
            for (a, b), (low, high) in new_bounds:
                bounds_lines.append(f"dir=({a:.4f}, {b:.4f})  [{low:.3f}, {high:.3f}]")
            bounds_view.setPlainText("\n".join(bounds_lines))

        def generate_and_plot():
            new_bounds = generate_bounds()

            self.__example_bounds = new_bounds
            latest_bounds.clear()
            latest_bounds.extend(new_bounds)
            update_bounds_view(new_bounds)

            plot = self.__primary_plot
            if plot is not None:
                if not self.__node_pair_bounds:
                    self.__register_node_pair(new_bounds)
                else:
                    self.__node_pair_bounds[0] = new_bounds
                for title, selection in self.__diagram_selections.items():
                    if 0 in selection:
                        target_plot = self.__plot_map.get(title)
                        if target_plot is not None:
                            self.__render_plot(target_plot)

        button_row = QWidget()
        button_layout = QHBoxLayout(button_row)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(6)
        button_layout.addStretch(1)
        generate_button = QPushButton("Generate & Plot")
        generate_button.clicked.connect(generate_and_plot)
        button_layout.addWidget(generate_button)
        # Temporary: overlay current bounds on the primary diagram.
        overlay_button = QPushButton("Add Pair")
        def add_overlay():
            new_bounds = generate_bounds()
            update_bounds_view(new_bounds)
            latest_bounds.clear()
            latest_bounds.extend(new_bounds)
            pair_index = self.__register_node_pair(new_bounds)
            title = self.__primary_plot.plot_title
            self.__diagram_selections.setdefault(title, set()).add(pair_index)
            self.__render_plot(self.__primary_plot)
            self.__rebuild_diagram_groups()
        overlay_button.clicked.connect(add_overlay)
        button_layout.addWidget(overlay_button)
        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.close)
        button_layout.addWidget(close_button)

        dialog_layout.addWidget(limit_label)
        dialog_layout.addWidget(limit_spin)
        dialog_layout.addWidget(directions_label)
        dialog_layout.addWidget(directions_value)
        dialog_layout.addWidget(directions_slider)
        dialog_layout.addWidget(QLabel("Generated bounds"))
        dialog_layout.addWidget(bounds_view)
        dialog_layout.addWidget(button_row)

        generate_and_plot()
        dialog.resize(360, 380)
        dialog.show()

    def __on_limits_changed(self, source: PlotWidget):
        if self.__syncing or not getattr(source, "locked", False):
            return
        ax = source.axes
        if ax is None:
            return
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        self.__syncing = True
        for widget in self.plots:
            if widget is source or not getattr(widget, "locked", False):
                continue
            target_ax = widget.axes
            target_canvas = widget.canvas
            if target_ax is None or target_canvas is None:
                continue
            target_ax.set_xlim(xlim)
            target_ax.set_ylim(ylim)
            target_canvas.draw_idle()
        self.__syncing = False

    def eventFilter(self, watched, event):
        if watched is self.__scroll_area.viewport() and event.type() == QEvent.Type.Resize:
            self.__relayout_plots()
        return super().eventFilter(watched, event)
