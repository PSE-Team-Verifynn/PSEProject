from typing import List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QScrollArea,
    QSizePolicy,
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
    __scale_list: QListWidget
    __syncing: bool

    def __init__(self, configuration: DiagramConfig):
        self.__syncing = False
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

        grid_host = QWidget()
        self.__plot_grid = QGridLayout(grid_host)
        self.__plot_grid.setContentsMargins(4, 4, 4, 4)
        self.__plot_grid.setSpacing(12)

        # Placeholder plot cards (2x2 grid).
        self.plots = []
        self.__plot_map = {}
        for i in range(4):
            title = f"Diagram {i + 1:02d}"
            plot_card = self.__create_plot_card(title)
            self.plots.append(plot_card)
            self.__plot_map[title] = plot_card
            row = i // 2
            col = i % 2
            self.__plot_grid.addWidget(plot_card, row, col)

        self.__plot_grid.setColumnStretch(0, 1)
        self.__plot_grid.setColumnStretch(1, 1)

        scroll_area.setWidget(grid_host)
        layout.addWidget(scroll_area)

        return container

    def get_side_bar(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        title = QLabel("Settings")
        title.setObjectName("title")
        layout.addWidget(title)

        for diagram_index in range(1, 5):
            group = QGroupBox(f"Diagram {diagram_index:02d}")
            group.setObjectName("card")
            group_layout = QVBoxLayout(group)
            group_layout.setContentsMargins(6, 20, 6, 6)
            group_layout.setSpacing(0)
            for pair_index in range(1, 4):
                check_box = QCheckBox(f"Node Pair {pair_index}")
                check_box.setObjectName("transparent-button")
                group_layout.addWidget(check_box)
            layout.addWidget(group)

        scale_group = QGroupBox("Shared Scale")
        scale_layout = QVBoxLayout(scale_group)
        scale_layout.setContentsMargins(6, 6, 6, 6)
        scale_layout.setSpacing(4)
        self.__scale_list = QListWidget()
        self.__scale_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.__scale_list.addItems(["Diagram 01", "Diagram 02", "Diagram 03", "Diagram 04"])
        scale_layout.addWidget(self.__scale_list)
        apply_scale_button = QPushButton("Apply Shared Scale")
        apply_scale_button.clicked.connect(self.__apply_shared_scale)
        scale_layout.addWidget(apply_scale_button)
        layout.addWidget(scale_group)

        node_pairs_group = QGroupBox("Node Pairs")
        node_pairs_layout = QVBoxLayout(node_pairs_group)
        node_pairs_layout.setContentsMargins(6, 6, 6, 6)
        node_pairs_layout.setSpacing(4)
        node_pairs_list = QListWidget()
        node_pairs_list.addItems(["Node 1", "Node 2", "Node 3"])
        node_pairs_layout.addWidget(node_pairs_list)
        layout.addWidget(node_pairs_group)

        layout.addStretch(1)

        edit_button = QPushButton("Edit Comparison")
        layout.addWidget(edit_button, alignment=Qt.AlignmentFlag.AlignLeft)

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

    def __create_plot_card(self, title: str) -> PlotWidget:
        card = PlotWidget()
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(8, 8, 8, 8)
        card_layout.setSpacing(6)

        plot_placeholder = QFrame()
        plot_placeholder.setFrameShape(QFrame.Shape.StyledPanel)
        plot_placeholder.setMinimumSize(520, 360)
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
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.set_aspect("auto")
        ax.set_xlim(-4, 3)
        ax.set_ylim(-2, 4.5)

        poly_points = np.array(
            [
                [-2.6, 3.9],
                [-1.4, 3.6],
                [-0.3, 2.1],
                [0.7, 1.5],
                [0.8, -1.5],
                [0.1, -1.7],
                [-1.2, -0.2],
                [-2.2, 1.1],
            ]
        )
        polygon = Polygon(poly_points, closed=True, facecolor="#59aef2", edgecolor="#3b6ea8", alpha=0.7)
        ax.add_patch(polygon)

        rng = np.random.default_rng(7)
        xs = rng.normal(-0.8, 0.35, 160)
        ys = rng.normal(1.5, 0.85, 160)
        ax.scatter(xs, ys, s=18, color="#d26b3a", edgecolor="#6b2f17", alpha=0.9)

        plot_layout.addWidget(toolbar)
        plot_layout.addWidget(canvas, stretch=1)
        card_layout.addWidget(plot_placeholder)

        footer = QWidget()
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

    def __toggle_lock(self, widget: PlotWidget, button: QPushButton):
        widget.locked = not getattr(widget, "locked", False)
        button.setText("Unlock" if widget.locked else "Lock")

    def __apply_shared_scale(self):
        selected_titles = [item.text() for item in self.__scale_list.selectedItems()]
        if selected_titles:
            widgets = [self.__plot_map[title] for title in selected_titles if title in self.__plot_map]
        else:
            widgets = list(self.plots)

        axes_list = [w.axes for w in widgets if getattr(w, "axes", None) is not None]
        if not axes_list:
            return

        xmins, xmaxs, ymins, ymaxs = [], [], [], []
        for ax in axes_list:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            xmins.append(xlim[0])
            xmaxs.append(xlim[1])
            ymins.append(ylim[0])
            ymaxs.append(ylim[1])

        shared_xlim = (min(xmins), max(xmaxs))
        shared_ylim = (min(ymins), max(ymaxs))

        self.__syncing = True
        for widget in widgets:
            ax = widget.axes
            canvas = widget.canvas
            if ax is None or canvas is None:
                continue
            ax.set_xlim(shared_xlim)
            ax.set_ylim(shared_ylim)
            canvas.draw_idle()
        self.__syncing = False

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
