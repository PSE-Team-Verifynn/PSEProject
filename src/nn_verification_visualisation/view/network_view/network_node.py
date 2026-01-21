from typing import Callable

from PySide6.QtGui import QBrush, QColor, QPen
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsItem
from PySide6.QtCore import Qt


class NetworkNode(QGraphicsEllipseItem):
    selectable: bool
    index: int
    layer_index: int
    on_click: Callable[[tuple[int, int]], None]

    def __init__(self, index: int, layer_index: int, radius: float, on_click: Callable[[tuple[int, int]], None], selectable: bool):
        super().__init__(-radius, -radius, radius * 2, radius * 2)
        self.index = index
        self.layer_index = layer_index
        self.on_click = on_click

        # Visual setup
        self.setBrush(QBrush(QColor("white")))
        self.setPen(QPen(Qt.GlobalColor.black, 2))

        # Selection flags
        if selectable:
            self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
            self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsFocusable)

    def mousePressEvent(self, event):
        self.setSelected(not self.isSelected())
        if self.on_click:
            self.on_click((self.layer_index, self.index))
        event.accept()

    def paint(self, painter, option, widget=None):
        # Override paint to customize appearance when selected
        painter.setRenderHint(painter.RenderHint.Antialiasing)

        # Change color if selected
        if self.isSelected():
            painter.setBrush(QBrush(QColor("#4CAF50")))  # Green when selected
        else:
            painter.setBrush(self.brush())

        painter.setPen(self.pen())
        painter.drawEllipse(self.rect())