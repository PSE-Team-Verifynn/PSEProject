from typing import List

from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF, QLinearGradient, QBrush
from PySide6.QtWidgets import QGraphicsItem, QStyleOptionGraphicsItem, QWidget
from PySide6.QtCore import QRectF, QLineF, Qt

from nn_verification_visualisation.view.network_view.network_node import NetworkNode


class NetworkEdgeBatch(QGraphicsItem):
    def __init__(self, source_nodes, target_nodes, force_block=False):
        super().__init__()
        self.source_nodes = source_nodes
        self.target_nodes = target_nodes
        self.use_block_mode = force_block

        if self.use_block_mode:
            self.__init_block_geometry()
        else:
            self.__init_line_geometry()

        self.setZValue(-1)

    def __init_block_geometry(self):
        s_top = self.source_nodes[0].scenePos()
        s_bot = self.source_nodes[-1].scenePos()
        t_top = self.target_nodes[0].scenePos()
        t_bot = self.target_nodes[-1].scenePos()

        self.polygon = QPolygonF([s_top, t_top, t_bot, s_bot])
        self._bounding_rect = self.polygon.boundingRect()

    def __init_line_geometry(self):
        self.lines = [QLineF(s.scenePos(), t.scenePos())
                      for s in self.source_nodes
                      for t in self.target_nodes]

        # Calculate bounding box from layer positions
        self._bounding_rect = QRectF(
            self.source_nodes[0].scenePos().x(),
            min(self.source_nodes[0].scenePos().y(), self.target_nodes[0].scenePos().y()),
            self.target_nodes[0].scenePos().x() - self.source_nodes[0].scenePos().x(),
            max(self.source_nodes[-1].scenePos().y(), self.target_nodes[-1].scenePos().y())
        )

    def boundingRect(self):
        return self._bounding_rect

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget | None = None) -> None:
        if self.use_block_mode:
            # Draw a subtle gradient block
            gradient = QLinearGradient(self._bounding_rect.topLeft(), self._bounding_rect.topRight())
            gradient.setColorAt(0, QColor(100, 100, 100, 40))  # Very light at start
            gradient.setColorAt(1, QColor(100, 100, 100, 80))  # Slightly darker at end

            painter.setBrush(QBrush(gradient))
            painter.setPen(Qt.NoPen)
            painter.drawPolygon(self.polygon)
        else:
            # Traditional line rendering
            pen = QPen(QColor(150, 150, 150, 100))
            pen.setWidth(20)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.drawLines(self.lines)