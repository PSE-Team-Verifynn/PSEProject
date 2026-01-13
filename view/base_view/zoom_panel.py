from typing import Callable

from PySide6.QtWidgets import QWidget

class ZoomPanel(QWidget):
    on_zoom_changed: Callable[[float], None]