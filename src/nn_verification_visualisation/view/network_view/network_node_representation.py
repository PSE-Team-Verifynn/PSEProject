from typing import Callable

from PySide6.QtGui import QBrush, QColor, QPen, QTransform
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsItem, QStyleOptionGraphicsItem, QGraphicsLineItem
from PySide6.QtCore import Qt


class NetworkLayerLine(QGraphicsLineItem):
    """
    Visual proxy for a layer when the network is zoomed out in performance mode.
    """

    def __init__(self, x_pos: float, top_y: float, bot_y: float):
        super().__init__()

        self.setLine(x_pos, top_y, x_pos, bot_y)

        # Style the line to look like a "spine" for the nodes
        pen = QPen(QColor(80, 80, 80))
        pen.setWidth(3)
        pen.setCosmetic(True)  # width stays constant on screen
        self.setPen(pen)
        self.setZValue(0)  # Behind nodes (Z=1), above edges (Z=-1)


class NetworkNode(QGraphicsEllipseItem):
    selectable: bool
    index: int
    layer_index: int
    on_click: Callable[[tuple[int, int]], None]

    # Visual setup variables
    color_unselected: QColor = QColor("black")
    color_selected: QColor = QColor("#4CAF50")
    color_node_outline: QColor = QColor("black")
    outline_pen_thickness: int = 2

    # LOD Settings
    lod_enabled: bool = False
    lod_threshold: float = 0.0  # Scale threshold below which nodes disappear

    def __init__(self, index: int, layer_index: int, radius: float, on_click: Callable[[tuple[int, int]], None],
                 selectable: bool):
        super().__init__(-radius, -radius, radius * 2, radius * 2)
        self.index = index
        self.layer_index = layer_index
        self.on_click = on_click
        self.selectable = selectable

        self.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)

        self.setBrush(QBrush(self.color_unselected))
        pen = QPen(self.color_node_outline, self.outline_pen_thickness)
        pen.setCosmetic(True)
        self.setPen(pen)

        if self.selectable:
            self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
            self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsFocusable)

    def set_lod_mode(self, enabled: bool):
        self.lod_enabled = enabled

    def _is_too_small(self):
        """Checks if the item is too small on screen based on current transform."""
        if not self.lod_enabled or not self.scene() or not self.scene().views():
            return False

        # We estimate scale based on the first view's transform
        # (Assuming uniform scaling in x and y)
        view = self.scene().views()[0]
        scale = view.transform().m11()
        return scale < self.lod_threshold

    def mousePressEvent(self, event):
        # Disable interaction if zoomed out in LOD mode
        if self._is_too_small():
            event.ignore()
            return

        if self.selectable:
            self.setSelected(not self.isSelected())
        if self.on_click:
            self.on_click((self.layer_index, self.index))
        event.accept()

    def paint(self, painter, option: QStyleOptionGraphicsItem, widget=None):
        # LOD Check: If enabled and too small, do not draw the ellipse
        if self.lod_enabled:
            # option.levelOfDetailFromTransform gives a rough estimate of scale
            lod = option.levelOfDetailFromTransform(painter.worldTransform())
            if lod < self.lod_threshold:
                return

        painter.setRenderHint(painter.RenderHint.Antialiasing)

        if self.isSelected():
            painter.setBrush(QBrush(self.color_selected))
        else:
            painter.setBrush(self.brush())

        painter.setPen(self.pen())
        painter.drawEllipse(self.rect())