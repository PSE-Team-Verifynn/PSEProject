from typing import Callable

from PySide6.QtGui import QBrush, QColor, QPen
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsItem
from PySide6.QtCore import Qt


class NetworkNode(QGraphicsEllipseItem):
    selectable: bool
    index: int
    layer_index: int
    on_click: Callable[[tuple[int, int]], None]

    #temporary setup variables
    color_unselected: QColor = QColor("black")
    color_selected: QColor = QColor("#4CAF50")
    color_node_outline : QColor = QColor("black")
    outline_pen_thickness : int = 2

    def __init__(self, index: int, layer_index: int, radius: float, on_click: Callable[[tuple[int, int]], None], selectable: bool):
        super().__init__(-radius, -radius, radius * 2, radius * 2)
        self.index = index
        self.layer_index = layer_index
        self.on_click = on_click
        self.selectable = selectable

        self.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)

        # Visual setup
        self.setBrush(QBrush(self.color_unselected))
        pen = QPen(self.color_node_outline, self.outline_pen_thickness)
        pen.setCosmetic(True)
        self.setPen(pen)

        # Selection flags
        if self.selectable:
            self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
            self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsFocusable)

    def mousePressEvent(self, event):
        if self.selectable:
            self.setSelected(not self.isSelected())
        if self.on_click:
            self.on_click((self.layer_index, self.index))
        event.accept()

    def paint(self, painter, option, widget=None):
        # Override paint to customize appearance when selected
        painter.setRenderHint(painter.RenderHint.Antialiasing)

        # Change color if selected
        if self.isSelected():
            painter.setBrush(QBrush(self.color_selected))  # Green when selected
        else:
            painter.setBrush(self.brush())

        painter.setPen(self.pen())
        painter.drawEllipse(self.rect())