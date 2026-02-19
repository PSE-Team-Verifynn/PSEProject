from __future__ import annotations

from typing import Callable

import numpy as np
from PySide6.QtGui import QColor
from matplotlib.axes import Axes
from matplotlib.patches import Polygon
from PySide6.QtWidgets import QDialog, QVBoxLayout, QPushButton

from nn_verification_visualisation.view.plot_view.plot_widget import PlotWidget


class PlotWidget2D(PlotWidget):
    """
    Renders flat 2-D polytope polygons using Sutherland-Hodgman clipping.
    Syncs pan/zoom between locked peers via xlim/ylim callbacks.
    """

    def __init__(
        self,
        on_limits_changed: Callable[[PlotWidget], None],
        title: str = "",
        parent=None,
    ):
        super().__init__(on_limits_changed, title, parent)

    # ---------------------------------------------------------------- axes setup --

    def _setup_axes(self) -> None:
        """Create a standard 2-D axes."""
        ax: Axes = self.figure.add_subplot(111)
        ax.set_title(self.title, fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.set_aspect("auto")
        self.axes = ax

    # -------------------------------------------------------------- fullscreen --

    def fullscreen(self) -> None:
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
            self.plot_layout.insertWidget(0, self.toolbar)
            self.plot_layout.addWidget(self.canvas, stretch=1)

        dialog.finished.connect(restore)
        dialog.showMaximized()

    # --------------------------------------------------------------- rendering --

    def render_plot(
        self,
        polygons: list[list[tuple[float, float]]],
        colors: list[QColor],
        polygon_names: list[str],
    ) -> None:
        if self.axes is None or self.canvas is None:
            return

        if polygons is None:
            polygons = []

        self.axes.cla()
        self.axes.grid(True, alpha=0.2)
        self.axes.set_title(self.title, fontsize=9)
        self.axes.set_aspect("auto")
        self.title_widget.setText(self.title)

        all_points: list[tuple[float, float]] = []
        legend_handles = []
        legend_labels = []

        while len(colors) < len(polygons):
            colors.append(QColor("#000000"))

        for index, polygon_points in enumerate(polygons):
            if not polygon_points or len(polygon_points) < 3:
                continue

            face_color = colors[index]
            edge_color = face_color.darker(150)
            all_points.extend(polygon_points)
            poly_array = np.array(polygon_points)
            patch = Polygon(
                poly_array,
                closed=True,
                facecolor=face_color.getRgbF(),
                edgecolor=edge_color.getRgbF(),
                alpha=0.6,
            )
            self.axes.add_patch(patch)
            legend_handles.append(patch)
            legend_labels.append(
                polygon_names[index] if polygon_names[index] is not None else ""
            )

        if all_points:
            xs = [p[0] for p in all_points]
            ys = [p[1] for p in all_points]
            self.axes.set_xlim(min(xs) - 0.5, max(xs) + 0.5)
            self.axes.set_ylim(min(ys) - 0.5, max(ys) + 0.5)

        self.axes.legend(
            legend_handles, legend_labels,
            loc="upper right", fontsize=7, frameon=True,
        )
        # Attach callbacks AFTER all limits are set so xlim_changed/ylim_changed
        # don't fire mid-render and propagate stale state to locked peers.
        self._attach_callbacks()
        self.canvas.draw_idle()

    # ------------------------------------------ view-sync public API --

    def get_view_state(self) -> dict:
        return {
            "xlim": self.axes.get_xlim(),
            "ylim": self.axes.get_ylim(),
        }

    def apply_view_state(self, state: dict) -> None:
        # Reject states from a 3-D source: locking across dimensions is
        # meaningless (elev/azim/zlim have no 2-D equivalent) and the
        # accidental xlim/ylim bleed-through from 3-D state is misleading.
        if self._syncing or "xlim" not in state or "elev" in state:
            return
        self._syncing = True
        try:
            self.axes.set_xlim(state["xlim"])
            self.axes.set_ylim(state["ylim"])
            self.canvas.draw_idle()
        finally:
            self._syncing = False

    # ------------------------------------------------------- callbacks --

    def _attach_callbacks(self) -> None:
        self._disconnect_callbacks()
        cids = [
            self.axes.callbacks.connect(
                "xlim_changed", lambda _ax: self._on_limits_changed(self)
            ),
            self.axes.callbacks.connect(
                "ylim_changed", lambda _ax: self._on_limits_changed(self)
            ),
        ]
        self.limit_callback_ids = cids