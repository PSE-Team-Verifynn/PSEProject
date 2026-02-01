from typing import List

from PySide6.QtCore import Qt, QEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QDialog,
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
from nn_verification_visualisation.controller.input_manager.plot_view_controller import PlotViewController
from nn_verification_visualisation.view.plot_view.comparison_loading_widget import ComparisonLoadingWidget
from nn_verification_visualisation.view.plot_view.plot_widget import PlotWidget
from nn_verification_visualisation.view.base_view.tab import Tab

class PlotPage(Tab):
    controller: PlotViewController
    configuration: DiagramConfig
    plots: List[PlotWidget]
    locked: List[PlotWidget]
    __plot_grid: QGridLayout
    __plot_map: dict[str, PlotWidget]
    __syncing: bool
    __scroll_area: QScrollArea | None
    __grid_host: QWidget | None
    __bottom_spacer_height: int
    __diagram_groups_layout: QVBoxLayout | None
    __node_pairs_list: QListWidget | None
    __node_pairs_layout: QVBoxLayout | None

    def __init__(self, controller: PlotViewController):


        self.__syncing = False
        self.__scroll_area = None
        self.__grid_host = None
        self.__bottom_spacer_height = 32
        self.plots = []
        self.__plot_map = {}
        self.__diagram_groups_layout = None
        self.__node_pairs_list = None
        self.__node_pairs_layout = None
        self.controller = controller
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

        plot_card = self.__create_plot_card("Diagram 01")
        self.plots.append(plot_card)
        self.__plot_map["Diagram 01"] = plot_card
        self.controller.register_plot("Diagram 01")
        self.__render_plot(plot_card)
        self.__rebuild_diagram_groups()

        scroll_area.setWidget(grid_host)
        layout.addWidget(scroll_area)

        scroll_area.viewport().installEventFilter(self)
        self.__relayout_plots()
        return container
        # return ComparisonLoadingWidget()

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
        size_slider.setValue(self.controller.card_size)
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

        return container

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

    def __create_plot_card(
        self,
        title: str,
    ) -> PlotWidget:
        card = PlotWidget()
        card.plot_title = title
        card.setFixedSize(self.controller.card_size, self.controller.card_size)
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

        card.limit_callback_ids = None

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
        columns = max(1, (available_width + spacing) // (self.controller.card_size + spacing))
        required_width = (
            columns * self.controller.card_size
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
        self.controller.set_card_size(value)
        for widget in self.plots:
            widget.setFixedSize(self.controller.card_size, self.controller.card_size)
        self.__relayout_plots()

    def __register_node_pair(
        self, bounds: list[tuple[tuple[float, float], tuple[float, float]]]
    ) -> int:
        pair_index = self.controller.add_node_pair(bounds)
        self.__refresh_node_pairs_list()
        self.__rebuild_diagram_groups()
        return pair_index

    def __refresh_node_pairs_list(self):
        if self.__node_pairs_list is None:
            return
        self.__node_pairs_list.clear()
        self.__node_pairs_list.addItems(self.controller.get_node_pairs())

    def __remove_selected_pair(self):
        if self.__node_pairs_list is None:
            return
        row = self.__node_pairs_list.currentRow()
        self.controller.remove_node_pair(row)
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
            group = QGroupBox()
            group.setObjectName("card")
            group_layout = QVBoxLayout(group)
            group_layout.setContentsMargins(6, 6, 6, 6)
            group_layout.setSpacing(0)

            header = QWidget()
            header_layout = QHBoxLayout(header)
            header_layout.setContentsMargins(0, 0, 0, 0)
            header_layout.setSpacing(6)
            header_layout.addWidget(QLabel(title))
            header_layout.addStretch(1)
            delete_button = QPushButton("🗑")
            delete_button.setFixedSize(28, 28)
            delete_button.clicked.connect(lambda _=False, t=title: self.__remove_diagram(t))
            header_layout.addWidget(delete_button)
            group_layout.addWidget(header)
            selection = self.controller.get_selection(title)
            for idx, pair_name in enumerate(self.controller.get_node_pairs()):
                check_box = QCheckBox(pair_name)
                check_box.setChecked(idx in selection)
                check_box.stateChanged.connect(
                    lambda state, t=title, i=idx: self.__on_pair_toggled(t, i, state)
                )
                group_layout.addWidget(check_box)
            self.__diagram_groups_layout.addWidget(group)

    def __on_pair_toggled(self, plot_title: str, pair_index: int, state: int):
        self.controller.change_plot(
            plot_title,
            state == Qt.CheckState.Checked.value,
            pair_index,
        )
        plot = self.__plot_map.get(plot_title)
        if plot is not None:
            self.__render_plot(plot)

    def __remove_diagram(self, title: str):
        plot = self.__plot_map.get(title)
        if plot is None:
            return
        if plot in self.plots:
            self.plots.remove(plot)
        self.__plot_map.pop(title, None)
        self.controller.remove_plot(title)
        plot.setParent(None)
        plot.deleteLater()
        self.__rebuild_diagram_groups()
        self.__relayout_plots()

    def __render_plot(self, plot: PlotWidget):
        if plot.axes is None or plot.canvas is None:
            return
        ax = plot.axes
        ax.cla()
        self.__attach_limit_callbacks(plot)
        ax.grid(True, alpha=0.2)
        title = getattr(plot, "plot_title", "Diagram")
        ax.set_title(title, fontsize=9)
        selection = self.controller.get_selection(title)
        all_points: list[tuple[float, float]] = []
        legend_handles = []
        legend_labels = []
        for order, pair_index in enumerate(sorted(selection)):
            if pair_index >= len(self.controller.node_pair_bounds):
                continue
            bounds = self.controller.get_node_pair_bounds(pair_index)
            poly_points = self.controller.compute_polygon(bounds)
            if not poly_points:
                continue
            all_points.extend(poly_points)
            face, edge = self.controller.get_node_pair_colors(pair_index)
            poly_array = np.array(poly_points)
            polygon = Polygon(poly_array, closed=True, facecolor=face, edgecolor=edge, alpha=0.6)
            ax.add_patch(polygon)
            legend_handles.append(polygon)
            legend_labels.append(self.controller.get_node_pairs()[pair_index])
        if all_points:
            xs = [p[0] for p in all_points]
            ys = [p[1] for p in all_points]
            ax.set_xlim(min(xs) - 0.5, max(xs) + 0.5)
            ax.set_ylim(min(ys) - 0.5, max(ys) + 0.5)
        if legend_handles:
            ax.legend(legend_handles, legend_labels, loc="upper right", fontsize=7, frameon=True)
        plot.canvas.draw_idle()

    def __attach_limit_callbacks(self, plot: PlotWidget):
        ax = plot.axes
        if ax is None:
            return
        existing = getattr(plot, "limit_callback_ids", None)
        if existing:
            for cid in existing:
                try:
                    ax.callbacks.disconnect(cid)
                except Exception:
                    pass
        cids = (
            ax.callbacks.connect("xlim_changed", lambda _ax: self.__on_limits_changed(plot)),
            ax.callbacks.connect("ylim_changed", lambda _ax: self.__on_limits_changed(plot)),
        )
        plot.limit_callback_ids = cids

    def __add_diagram_from_current_bounds(self):
        title = f"Diagram {len(self.plots) + 1:02d}"
        plot_card = self.__create_plot_card(title)
        self.plots.append(plot_card)
        self.__plot_map[title] = plot_card
        self.controller.register_plot(title)
        self.__render_plot(plot_card)
        self.__rebuild_diagram_groups()
        self.__relayout_plots()

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
