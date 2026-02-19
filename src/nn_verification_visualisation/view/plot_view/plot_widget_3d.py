from __future__ import annotations

from typing import Callable

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QWidget,
)
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 – registers '3d' projection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from nn_verification_visualisation.view.plot_view.plot_widget import PlotWidget

# ---------------------------------------------------------------------------
# Camera presets: (elev°, azim°)
# ---------------------------------------------------------------------------
_PRESET_DEFAULT   = (25.0, -60.0)
_PRESET_ISOMETRIC = (25.0, -60.0)
_PRESET_FRONT     = ( 0.0, -90.0)   # XZ plane — Neuron 1 vs 3
_PRESET_SIDE      = ( 0.0,   0.0)   # YZ plane — Neuron 2 vs 3
_PRESET_TOP       = (90.0, -90.0)   # XY plane — Neuron 1 vs 2

# Elevation clamp: prevents upside-down flips and degenerate flat views
_ELEV_MIN = 5.0
_ELEV_MAX = 85.0

# Type aliases
_Face3D     = list[tuple[float, float, float]]
_Polyhedron = list[_Face3D]


class PlotWidget3D(PlotWidget):
    """
    Renders 3-D convex polyhedra using Poly3DCollection.
    Provides elevation clamping, preset-view buttons, camera preservation
    across re-renders, and locked camera-angle sync between peers.
    """

    axes: Axes3D

    def __init__(
        self,
        on_limits_changed: Callable[[PlotWidget], None],
        title: str = "",
        parent=None,
    ):
        self._3d_toolbar_row: QWidget | None = None
        super().__init__(on_limits_changed, title, parent)
        # Insert the preset toolbar row after the base __init__ has built the
        # plot_layout (toolbar at 0, canvas at 1).
        self._insert_3d_toolbar()

    # ---------------------------------------------------------------- axes setup --

    def _setup_axes(self) -> None:
        """Create a 3-D axes and set the default camera angle."""
        ax: Axes3D = self.figure.add_subplot(111, projection="3d")
        ax.set_title(self.title, fontsize=9)
        self.axes = ax
        elev, azim = _PRESET_DEFAULT
        ax.view_init(elev=elev, azim=azim)

    # --------------------------------------------------------- 3-D toolbar --

    def _insert_3d_toolbar(self) -> None:
        """Build the preset-view row and insert it at index 1 (below the mpl toolbar)."""
        row = QWidget()
        row.setFixedHeight(26)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(2, 0, 2, 0)
        layout.setSpacing(3)

        label = QLabel("View:")
        label.setStyleSheet("font-size: 10px;")
        layout.addWidget(label)

        presets: list[tuple[str, tuple[float, float], str]] = [
            ("Iso",   _PRESET_ISOMETRIC, "Isometric view"),
            ("Front", _PRESET_FRONT,     "Front view — XZ plane (Neuron 1 vs 3)"),
            ("Side",  _PRESET_SIDE,      "Side view — YZ plane (Neuron 2 vs 3)"),
            ("Top",   _PRESET_TOP,       "Top view — XY plane (Neuron 1 vs 2)"),
        ]
        for btn_label, preset, tooltip in presets:
            btn = QPushButton(btn_label)
            btn.setToolTip(tooltip)
            btn.setFixedHeight(22)
            btn.setStyleSheet("font-size: 10px; padding: 0 6px;")
            # Default-argument capture avoids the late-binding closure trap
            btn.clicked.connect(
                lambda _checked=False, p=preset: self._apply_camera_preset(p)
            )
            layout.addWidget(btn)

        layout.addStretch(1)

        reset_btn = QPushButton("↺ Reset")
        reset_btn.setToolTip("Reset camera to default isometric angle")
        reset_btn.setFixedHeight(22)
        reset_btn.setStyleSheet("font-size: 10px; padding: 0 6px;")
        reset_btn.clicked.connect(
            lambda: self._apply_camera_preset(_PRESET_DEFAULT)
        )
        layout.addWidget(reset_btn)

        self._3d_toolbar_row = row
        # plot_layout: index 0 = mpl toolbar, 1 = this row, 2 = canvas
        self.plot_layout.insertWidget(1, row)

    # --------------------------------------------------------- camera control --

    def _apply_camera_preset(
        self,
        preset: tuple[float, float],
        draw: bool = True,
    ) -> None:
        """Snap the camera to (elev, azim) and optionally propagate to locked peers."""
        elev, azim = preset
        self.axes.view_init(elev=elev, azim=azim)
        if draw:
            self.canvas.draw_idle()
            if not self._syncing:
                self._on_limits_changed(self)

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
        if self._3d_toolbar_row is not None:
            self.plot_layout.removeWidget(self._3d_toolbar_row)
        self.plot_layout.removeWidget(self.canvas)

        self.toolbar.setParent(dialog)
        dialog_layout.addWidget(self.toolbar)
        if self._3d_toolbar_row is not None:
            self._3d_toolbar_row.setParent(dialog)
            dialog_layout.addWidget(self._3d_toolbar_row)
        self.canvas.setParent(dialog)
        dialog_layout.addWidget(self.canvas, stretch=1)

        def restore():
            dialog_layout.removeWidget(self.toolbar)
            if self._3d_toolbar_row is not None:
                dialog_layout.removeWidget(self._3d_toolbar_row)
            dialog_layout.removeWidget(self.canvas)

            self.toolbar.setParent(self)
            self.plot_layout.insertWidget(0, self.toolbar)
            if self._3d_toolbar_row is not None:
                self._3d_toolbar_row.setParent(self)
                self.plot_layout.insertWidget(1, self._3d_toolbar_row)
            self.canvas.setParent(self)
            self.plot_layout.addWidget(self.canvas, stretch=1)

        dialog.finished.connect(restore)
        dialog.showMaximized()

    # --------------------------------------------------------------- rendering --

    def render_plot(
        self,
        polyhedra: list[_Polyhedron],
        colors: list[QColor],
        polygon_names: list[str],
    ) -> None:
        """
        Render one or more convex polyhedra.

        *polyhedra*: list where each entry is a list of triangular faces
        (each face is a list of three (x, y, z) vertex tuples) as returned
        by PlotViewController.compute_polygon in 3-D mode.
        """
        if self.axes is None or self.canvas is None:
            return

        if polyhedra is None:
            polyhedra = []

        ax: Axes3D = self.axes

        # Preserve the user's camera angle across re-renders.
        # ax.cla() would reset it to matplotlib's defaults.
        saved_elev = ax.elev
        saved_azim = ax.azim

        ax.cla()
        ax.set_title(self.title, fontsize=9)
        self.title_widget.setText(self.title)

        while len(colors) < len(polyhedra):
            colors.append(QColor("#000000"))

        all_verts: list[tuple[float, float, float]] = []
        legend_proxies: list[Patch] = []
        legend_labels: list[str] = []

        for index, faces in enumerate(polyhedra):
            if not faces:
                continue

            face_color = colors[index]
            edge_color = face_color.darker(150)

            for face in faces:
                all_verts.extend(face)

            face_arrays = [np.array(face) for face in faces]
            fc_rgba = face_color.getRgbF()
            ec_rgba = edge_color.getRgbF()

            collection = Poly3DCollection(
                face_arrays,
                facecolor=(*fc_rgba[:3], 0.45),
                edgecolor=(*ec_rgba[:3], 0.8),
                linewidth=0.4,
            )
            ax.add_collection3d(collection)

            # Poly3DCollection is not a valid legend handle; use a proxy patch
            proxy = Patch(
                facecolor=(*fc_rgba[:3], 0.45),
                edgecolor=(*ec_rgba[:3], 0.8),
                label=polygon_names[index] if polygon_names[index] is not None else "",
            )
            legend_proxies.append(proxy)
            legend_labels.append(proxy.get_label())

        if all_verts:
            xs = [v[0] for v in all_verts]
            ys = [v[1] for v in all_verts]
            zs = [v[2] for v in all_verts]
            pad = 0.5
            ax.set_xlim(min(xs) - pad, max(xs) + pad)
            ax.set_ylim(min(ys) - pad, max(ys) + pad)
            ax.set_zlim(min(zs) - pad, max(zs) + pad)

        ax.set_xlabel("Neuron 1", fontsize=7)
        ax.set_ylabel("Neuron 2", fontsize=7)
        ax.set_zlabel("Neuron 3", fontsize=7)
        ax.tick_params(labelsize=6)

        if legend_proxies:
            ax.legend(
                legend_proxies, legend_labels,
                loc="upper right", fontsize=7, frameon=True,
            )

        # Restore the camera the user had before the re-render
        ax.view_init(elev=saved_elev, azim=saved_azim)

        self._attach_callbacks()
        self.canvas.draw_idle()

    # ------------------------------------------ view-sync public API --

    def get_view_state(self) -> dict:
        ax: Axes3D = self.axes
        return {
            "elev": ax.elev,
            "azim": ax.azim,
            "xlim": ax.get_xlim(),
            "ylim": ax.get_ylim(),
            "zlim": ax.get_zlim(),
        }

    def apply_view_state(self, state: dict) -> None:
        if self._syncing or "elev" not in state:
            return
        self._syncing = True
        try:
            ax: Axes3D = self.axes
            # Clamp on the receiving end too, so a peer that somehow exceeded
            # the range cannot drag this widget out of bounds
            elev = max(_ELEV_MIN, min(_ELEV_MAX, state["elev"]))
            ax.view_init(elev=elev, azim=state["azim"])
            ax.set_xlim(state["xlim"])
            ax.set_ylim(state["ylim"])
            ax.set_zlim(state["zlim"])
            self.canvas.draw_idle()
        finally:
            self._syncing = False

    # ------------------------------------------------------- callbacks --

    def _attach_callbacks(self) -> None:
        """
        Hook canvas events to detect both rotation (motion_notify_event) and
        home/back/forward button presses (draw_event).

        motion_notify_event: fires during mouse drag rotation. A zero-delay
        QTimer defers the work so matplotlib's handler updates ax.elev/azim first.

        draw_event: fires after every canvas.draw(), including the matplotlib
        toolbar home/back/forward buttons which restore saved views. We compare
        elev/azim/limits against the last known values to avoid feedback loops
        from our own draw_idle() calls.
        """
        self._disconnect_callbacks()

        # Snapshot the current camera so draw_event can detect real changes.
        self._last_elev = self.axes.elev
        self._last_azim = self.axes.azim

        motion_cid = self.canvas.mpl_connect(
            "motion_notify_event",
            lambda _evt: QTimer.singleShot(0, self._on_3d_view_changed),
        )
        draw_cid = self.canvas.mpl_connect(
            "draw_event",
            lambda _evt: self._on_3d_draw(),
        )
        self.limit_callback_ids = [motion_cid, draw_cid]

    def _on_3d_draw(self) -> None:
        """
        Fired after every canvas draw. Detects home/back/forward button
        actions by comparing current elev/azim against the last known values,
        then propagates if they changed.
        """
        if self._syncing:
            return
        ax: Axes3D = self.axes
        if ax.elev != self._last_elev or ax.azim != self._last_azim:
            self._last_elev = ax.elev
            self._last_azim = ax.azim
            self._on_limits_changed(self)

    def _on_3d_view_changed(self) -> None:
        """
        1. Clamp elevation to [_ELEV_MIN, _ELEV_MAX].
        2. Propagate the (possibly corrected) camera to locked peers.
        """
        if self._syncing:
            return

        ax: Axes3D = self.axes
        clamped_elev = max(_ELEV_MIN, min(_ELEV_MAX, ax.elev))
        if abs(clamped_elev - ax.elev) > 0.01:
            # Snap back before broadcasting so peers receive the clean angle
            ax.view_init(elev=clamped_elev, azim=ax.azim)
            self.canvas.draw_idle()

        self._on_limits_changed(self)