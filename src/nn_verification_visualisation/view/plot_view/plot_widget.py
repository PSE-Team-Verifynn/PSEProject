from __future__ import annotations

from typing import Callable, List

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QColor
from matplotlib.axes import Axes
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from nn_verification_visualisation.model.data.plot import Plot
from PySide6.QtWidgets import (
    QWidget, QDialog, QVBoxLayout, QSizePolicy,
    QFrame, QHBoxLayout, QLabel, QLayout, QPushButton,
)


class PlotWidget(QWidget):
    """
    Abstract base class for 2-D and 3-D plot cards.

    Responsibilities:
    - Qt card layout (frame, toolbar slot, canvas slot, footer)
    - Lock / fullscreen buttons
    - Shared state: title, locked, _syncing, limit_callback_ids
    - _on_limits_changed callback (accessible to subclasses)
    - _disconnect_callbacks / _rebuild_axes lifecycle helpers

    Subclasses must implement:
    - render_plot(polygons, colors, names)
    - get_view_state() -> dict
    - apply_view_state(state: dict)
    - fullscreen()
    - _attach_callbacks()          called at the end of each render
    - _setup_axes()                called once in __init__ to create self.axes
    """

    plot: Plot
    figure: Figure
    axes: Axes          # set by subclass via _setup_axes()
    canvas: FigureCanvas
    locked: bool
    toolbar: NavigationToolbar
    plot_layout: QLayout
    title: str
    _syncing: bool
    limit_callback_ids: List[int]

    # Single underscore so subclasses can call it
    _on_limits_changed: Callable[[PlotWidget], None]

    @staticmethod
    def make_plot_widget(
        on_limits_changed: Callable[[PlotWidget], None],
        title: str = "",
        parent=None,
        is_3d: bool = False,
    ) -> PlotWidget:
        if is_3d:
            from nn_verification_visualisation.view.plot_view.plot_widget_3d import PlotWidget3D
            return PlotWidget3D(on_limits_changed, title, parent)
        from nn_verification_visualisation.view.plot_view.plot_widget_2d import PlotWidget2D
        return PlotWidget2D(on_limits_changed, title, parent)

    def __init__(
        self,
        on_limits_changed: Callable[[PlotWidget], None],
        title: str = "",
        parent=None,
    ):
        super().__init__(parent)

        self.setObjectName("plot-card")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAutoFillBackground(True)

        self.title = title
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self._on_limits_changed = on_limits_changed
        self.limit_callback_ids = []
        self.locked = False
        self._syncing = False
        self.polygon_points = []

        # ---- card layout ----
        card_layout = QVBoxLayout()
        card_layout.setContentsMargins(8, 8, 8, 8)
        card_layout.setSpacing(6)

        plot_placeholder = QFrame()
        plot_placeholder.setObjectName("plot-container")
        plot_placeholder.setFrameShape(QFrame.Shape.StyledPanel)
        plot_placeholder.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        plot_layout = QVBoxLayout(plot_placeholder)
        plot_layout.setContentsMargins(4, 4, 4, 4)
        plot_layout.setSpacing(4)

        figure = Figure(figsize=(3.2, 2.4))
        canvas = FigureCanvas(figure)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        toolbar = NavigationToolbar(canvas, plot_placeholder)

        self.figure = figure
        self.canvas = canvas
        self.toolbar = toolbar
        self.plot_layout = plot_layout

        # Subclass creates self.axes via _setup_axes()
        self._setup_axes()

        plot_layout.addWidget(toolbar)
        plot_layout.addWidget(canvas, stretch=1)
        card_layout.addWidget(plot_placeholder)

        # ---- footer ----
        self.title_widget = QLabel(title)
        self.title_widget.setObjectName("heading")

        footer = QWidget()
        footer.setFixedHeight(40)
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(0, 0, 0, 0)
        footer_layout.addSpacing(10)
        footer_layout.addWidget(self.title_widget)
        footer_layout.addStretch(1)

        self._lock_button = QPushButton()
        self._lock_button.setObjectName("icon-button-tight")
        self._lock_button.setIcon(QIcon(":assets/icons/plot/unlocked.svg"))
        self._lock_button.clicked.connect(
            lambda: self._toggle_lock(self._lock_button)
        )
        footer_layout.addWidget(self._lock_button)

        fullscreen_button = QPushButton()
        fullscreen_button.setObjectName("icon-button-tight")
        fullscreen_button.setIcon(QIcon(":assets/icons/plot/fullscreen.svg"))
        fullscreen_button.clicked.connect(self.fullscreen)
        footer_layout.addWidget(fullscreen_button)

        card_layout.addWidget(footer)
        self.setLayout(card_layout)

        self._attach_name_change_callback()

    # ---------------------------------------------------------------- abstract interface --
    # Subclasses must override all of these.

    def _setup_axes(self) -> None:
        """Create self.axes on self.figure. Called once from __init__."""
        raise NotImplementedError

    def render_plot(
        self,
        polygons: list,
        colors: list[QColor],
        polygon_names: list[str],
    ) -> None:
        raise NotImplementedError

    def get_view_state(self) -> dict:
        raise NotImplementedError

    def apply_view_state(self, state: dict) -> None:
        raise NotImplementedError

    def fullscreen(self) -> None:
        raise NotImplementedError

    def _attach_callbacks(self) -> None:
        """Register axes/canvas callbacks after each render. Called by subclass."""
        raise NotImplementedError

    # ---------------------------------------------------------------- shared helpers --

    def _toggle_lock(self, lock_button: QPushButton) -> None:
        self.locked = not self.locked
        icon = (
            ":assets/icons/plot/locked.svg"
            if self.locked
            else ":assets/icons/plot/unlocked.svg"
        )
        lock_button.setIcon(QIcon(icon))

    def _disconnect_callbacks(self) -> None:
        """Disconnect all currently registered axes/canvas callbacks."""
        for cid in self.limit_callback_ids:
            try:
                self.canvas.mpl_disconnect(cid)
            except Exception:
                pass
            try:
                self.axes.callbacks.disconnect(cid)
            except Exception:
                pass
        self.limit_callback_ids = []

    def _rebuild_axes(self) -> None:
        """
        Clear the figure and recreate axes via _setup_axes().
        Subclass should call this when it needs a fresh axes object,
        then re-attach the title callback.
        """
        self._disconnect_callbacks()
        self.figure.clear()
        self._setup_axes()
        self._attach_name_change_callback()

    def _attach_name_change_callback(self) -> None:
        # matplotlib Artist callbacks receive the artist as their argument,
        # not the new value — read get_title() inside the callback instead.
        self.axes.title.add_callback(self._on_title_changed)

    def _on_title_changed(self, _artist) -> None:
        self.title = self.axes.get_title()