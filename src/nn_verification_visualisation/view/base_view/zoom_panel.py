from typing import Callable, Any

from PySide6.QtWidgets import QWidget

class ZoomPanel(QWidget):
    on_zoom_changed: Callable[[float], Any]