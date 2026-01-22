from typing import List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QScrollArea,
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

    def __init__(self, configuration: DiagramConfig):
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
        for i in range(4):
            plot_card = self.__create_plot_card(f"Diagram {i + 1:02d}")
            self.plots.append(plot_card)
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
        title.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(title)

        for diagram_index in range(1, 5):
            group = QGroupBox(f"Diagram {diagram_index:02d}")
            group_layout = QVBoxLayout(group)
            group_layout.setContentsMargins(6, 6, 6, 6)
            group_layout.setSpacing(4)
            for pair_index in range(1, 4):
                group_layout.addWidget(QCheckBox(f"Node Pair {pair_index}"))
            layout.addWidget(group)

        scale_group = QGroupBox("Shared Scale")
        scale_layout = QVBoxLayout(scale_group)
        scale_layout.setContentsMargins(6, 6, 6, 6)
        scale_layout.setSpacing(4)
        scale_list = QListWidget()
        scale_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        scale_list.addItems(["Diagram 01", "Diagram 02", "Diagram 03", "Diagram 04"])
        scale_layout.addWidget(scale_list)
        scale_layout.addWidget(QPushButton("Apply Shared Scale"))
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
        pass

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
        plot_placeholder.setMinimumSize(260, 200)
        plot_layout = QVBoxLayout(plot_placeholder)
        plot_layout.setContentsMargins(4, 4, 4, 4)
        plot_layout.setSpacing(4)

        figure = Figure(figsize=(3.2, 2.4))
        canvas = FigureCanvas(figure)
        toolbar = NavigationToolbar(canvas, plot_placeholder)

        ax = figure.add_subplot(111)
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.set_aspect("equal", "box")
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
        plot_layout.addWidget(canvas)
        card_layout.addWidget(plot_placeholder)

        footer = QWidget()
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(0, 0, 0, 0)
        footer_layout.setSpacing(6)

        footer_layout.addWidget(QLabel(title))
        footer_layout.addStretch(1)
        footer_layout.addWidget(QPushButton("Lock"))
        footer_layout.addWidget(QPushButton("Fullscreen"))
        footer_layout.addWidget(QPushButton("Export"))

        card_layout.addWidget(footer)
        return card
