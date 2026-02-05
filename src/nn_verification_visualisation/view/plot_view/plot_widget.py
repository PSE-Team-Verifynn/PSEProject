from __future__ import annotations

from typing import Callable, List

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QColor
from matplotlib.axes import Axes
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.patches import Polygon

from nn_verification_visualisation.model.data.plot import Plot
from PySide6.QtWidgets import QWidget, QDialog, QVBoxLayout, QSizePolicy, QFrame, QHBoxLayout, QLabel, QLayout, \
    QPushButton


class PlotWidget(QWidget):
    plot: Plot
    figure: Figure
    axes: Axes
    canvas: FigureCanvas
    locked: bool
    toolbar: NavigationToolbar
    plot_layout: QLayout
    title: str

    __on_limits_changed: Callable[[PlotWidget], None]
    limit_callback_ids: List[int]

    def __init__(self, on_limits_changed: Callable[[PlotWidget], None], title: str = "", parent=None):
        super().__init__(parent)

        self.setObjectName("plot-card")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAutoFillBackground(True)

        self.title = title
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self.__on_limits_changed = on_limits_changed
        self.limit_callback_ids = []


        card_layout = QVBoxLayout()
        card_layout.setContentsMargins(8, 8, 8, 8)
        card_layout.setSpacing(6)

        plot_placeholder = QFrame()
        plot_placeholder.setObjectName("plot-container")
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
        self.figure = figure
        self.axes = ax
        self.canvas = canvas
        self.toolbar = toolbar
        self.plot_layout = plot_layout
        self.locked = False
        self.polygon_points = []
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.set_aspect("auto")

        plot_layout.addWidget(toolbar)
        plot_layout.addWidget(canvas, stretch=1)
        card_layout.addWidget(plot_placeholder)

        self.title_widget = QLabel(title)
        self.title_widget.setObjectName("heading")

        footer = QWidget()
        footer.setFixedHeight(40)
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(0, 0, 0, 0)
        footer_layout.addSpacing(10)
        footer_layout.addWidget(self.title_widget)
        footer_layout.addStretch(1)

        lock_button = QPushButton()
        lock_button.setObjectName("icon-button-tight")
        lock_button.setIcon(QIcon(":assets/icons/plot/unlocked.svg"))
        lock_button.clicked.connect(lambda: self.__toggle_lock(lock_button))
        footer_layout.addWidget(lock_button)

        fullscreen_button = QPushButton()
        fullscreen_button.setObjectName("icon-button-tight")
        fullscreen_button.setIcon(QIcon(":assets/icons/plot/fullscreen.svg"))
        fullscreen_button.clicked.connect(lambda: self.fullscreen())
        footer_layout.addWidget(fullscreen_button)

        self.__attach_name_change_callback()

        card_layout.addWidget(footer)

        self.setLayout(card_layout)


    def fullscreen(self):
        if self.canvas is None or self.toolbar is None or self.plot_layout is None:
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Fullscreen Plot")
        dialog_layout = QVBoxLayout(dialog)
        dialog_layout.setContentsMargins(6, 6, 6, 6)
        dialog_layout.setSpacing(6)

        self.plot_layout.removeWidget(self.toolbar)
        self.plot_layout.removeWidget(self.canvas)
        self.toolbar.setParent(dialog)
        self.canvas.setParent(dialog)
        dialog_layout.addWidget(self.toolbar)
        dialog_layout.addWidget(self.canvas, stretch=1)

        def restore():
            dialog_layout.removeWidget(self.toolbar)
            dialog_layout.removeWidget(self.canvas)
            self.toolbar.setParent(self)
            self.canvas.setParent(self)
            self.plot_layout.addWidget(self.toolbar)
            self.plot_layout.addWidget(self.canvas, stretch=1)

        dialog.finished.connect(restore)
        dialog.showMaximized()


    def __toggle_lock(self, lock_button: QPushButton):
        self.locked = not self.locked
        if self.locked:
            lock_button.setIcon(QIcon(":assets/icons/plot/locked.svg"))
        else:
            lock_button.setIcon(QIcon(":assets/icons/plot/unlocked.svg"))

    def render_plot(self, polygons: list[list[tuple[float, float]]], colors: list[QColor], polygon_names: list[str]) -> None:
        if self.axes is None or self.canvas is None:
            return

        if polygons is None:
            polygons = []

        print(f"Given polygons: {polygons}")

        self.axes.cla()
        self.axes.grid(True, alpha=0.2)
        self.axes.set_title(self.title, fontsize=9)

        self.title_widget.setText(self.title)

        self.__attach_limit_callbacks()

        all_points: list[tuple[float, float]] = []
        legend_handles = []
        legend_labels = []

        for index, polygon_points in enumerate(polygons):
            if not polygon_points or len(polygon_points) < 3:
                continue
            if len(colors) < len(polygons):
                for i in range(len(polygons) - len(colors)):
                    colors.append("0x00000")

            face_color = colors[index]
            edge_color = face_color.darker(150)
            all_points.extend(polygon_points)
            poly_array = np.array(polygon_points)
            print(f"Adding polygon {index}, {polygon_points}")
            polygon = Polygon(poly_array, closed=True, facecolor=face_color.getRgbF(), edgecolor=edge_color.getRgbF(), alpha=0.6)
            self.axes.add_patch(polygon)
            legend_handles.append(polygon)
            legend_labels.append(polygon_names[index] if polygon_names[index] is not None else "")

        if len(all_points) != 0:
            xs = [p[0] for p in all_points]
            ys = [p[1] for p in all_points]

            self.axes.set_xlim(min(xs) - 0.5, max(xs) + 0.5)
            self.axes.set_ylim(min(ys) - 0.5, max(ys) + 0.5)

        self.axes.legend(legend_handles, legend_labels, loc="upper right", fontsize=7, frameon=True)

        self.canvas.draw_idle()

    def __attach_name_change_callback(self):
        self.axes.title.add_callback(self.__on_name_changed)

    def __on_name_changed(self, new_title):
        self.title = new_title

    def __attach_limit_callbacks(self):
        if self.axes is None:
            return
        if self.limit_callback_ids:
            for cid in self.limit_callback_ids:
                try:
                    self.axes.callbacks.disconnect(cid)
                except Exception:
                    pass
        cids = [
            self.axes.callbacks.connect("xlim_changed", lambda _ax: self.__on_limits_changed(self)),
            self.axes.callbacks.connect("ylim_changed", lambda _ax: self.__on_limits_changed(self)),
        ]
        self.limit_callback_ids = cids