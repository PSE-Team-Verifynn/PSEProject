from typing import Callable

from model.data.plot import Plot
from PySide6.QtWidgets import QWidget

class PlotWidget(QWidget):
    plot: Plot
    on_export: Callable[[PlotWidget], None]
    on_fullscreen: Callable[[PlotWidget], None]
    on_lock_change: Callable[[PlotWidget], None]
    on_transform: Callable[[PlotWidget, tuple[float, float, float, float]], None]