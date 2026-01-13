from typing import Callable

from PySide6.QtWidgets import QGraphicsEllipseItem

class NetworkNode(QGraphicsEllipseItem):
    selectable: bool
    index: int
    on_click: Callable[[], None]