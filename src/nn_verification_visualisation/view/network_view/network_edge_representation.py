from typing import List, Optional

from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF, QLinearGradient, QBrush
from PySide6.QtWidgets import QGraphicsItem, QStyleOptionGraphicsItem, QWidget
from PySide6.QtCore import QRectF, QLineF, Qt


class NetworkEdgeBatch(QGraphicsItem):
    def __init__(self, source_nodes, target_nodes, force_block=False, use_weighted=False, weights: Optional[List[List[float]]] = None):
        super().__init__()
        self.source_nodes = source_nodes
        self.target_nodes = target_nodes
        self.use_block_mode = force_block
        self.use_weighted_mode = use_weighted
        self.weights = weights

        # Pre-calculate weight range for normalization
        self.min_weight = None
        self.max_weight = None

        if self.use_block_mode:
            self.__init_block_geometry()
        else:
            self.__init_line_geometry()

        self.setZValue(-1)

    def __init_block_geometry(self):
        if not self.source_nodes or not self.target_nodes:
            self.polygon = QPolygonF()
            self._bounding_rect = QRectF()
            return

        s_top = self.source_nodes[0].scenePos()
        s_bot = self.source_nodes[-1].scenePos()
        t_top = self.target_nodes[0].scenePos()
        t_bot = self.target_nodes[-1].scenePos()

        self.polygon = QPolygonF([s_top, t_top, t_bot, s_bot])
        self._bounding_rect = self.polygon.boundingRect()

    def __init_line_geometry(self):
        self.lines_data = []

        xs = self.source_nodes[0].scenePos().x()
        xt = self.target_nodes[0].scenePos().x()
        min_y = min(self.source_nodes[0].scenePos().y(), self.target_nodes[0].scenePos().y())
        max_y = max(self.source_nodes[-1].scenePos().y(), self.target_nodes[-1].scenePos().y())

        num_source = len(self.source_nodes)
        num_target = len(self.target_nodes)

        # Determine the indexing strategy based on shape
        transpose_needed = False
        if self.weights is not None:
            shape = self.weights.shape
            # ONNX format is typically [input, output] = [source, target]
            if shape == (num_source, num_target):
                transpose_needed = False  # Standard format: weights[source, target]
            elif shape == (num_target, num_source):
                transpose_needed = True  # Transposed format: weights[target, source]

        # Build the lines and find weight range
        weight_values = []
        for i, s in enumerate(self.source_nodes):
            for j, t in enumerate(self.target_nodes):
                line = QLineF(s.scenePos(), t.scenePos())

                w = 1.0
                if self.weights is not None:
                    try:
                        val = self.weights[i, j] if not transpose_needed else self.weights[j, i]
                        w = float(val)
                        weight_values.append(abs(w))
                    except (IndexError, ValueError) as e:
                        w = 0.0

                self.lines_data.append((line, w))

        # Calculate weight range ONCE during initialization
        if weight_values:
            self.min_weight = min(weight_values)
            self.max_weight = max(weight_values)
        else:
            self.min_weight = 0.0
            self.max_weight = 1.0

        self._bounding_rect = QRectF(xs, min_y, xt - xs, max_y - min_y)

    def boundingRect(self):
        return self._bounding_rect

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget | None = None) -> None:
        if self.use_block_mode:
            gradient = QLinearGradient(self._bounding_rect.topLeft(), self._bounding_rect.topRight())
            gradient.setColorAt(0, QColor(100, 100, 100, 40))
            gradient.setColorAt(1, QColor(100, 100, 100, 80))

            painter.setBrush(QBrush(gradient))
            painter.setPen(Qt.NoPen)
            painter.drawPolygon(self.polygon)
        else:
            base_color = QColor(150, 150, 150, 150)
            pen = QPen(base_color)
            pen.setCosmetic(True)

            # Check if we HAVE weights AND they're valid
            has_valid_weights = (self.weights is not None and
                                 self.min_weight is not None and
                                 len(self.lines_data) > 0)

            if self.use_weighted_mode and has_valid_weights:
                # WEIGHTED MODE - vary thickness and opacity based on weights
                min_thickness = 2.3
                max_thickness = 6.0

                for line, weight in self.lines_data:
                    abs_weight = abs(weight)

                    # Normalize weight to 0-1 range based on pre-calculated range
                    if self.max_weight > self.min_weight:
                        normalized = (abs_weight - self.min_weight) / (self.max_weight - self.min_weight)
                    else:
                        normalized = 1.0 if abs_weight > 0 else 0.0

                    # Map to thickness
                    thickness = min_thickness + normalized * (max_thickness - min_thickness)
                    pen.setWidthF(thickness)

                    # Alpha based on normalized weight (stronger weights = more opaque)
                    alpha = int(80 + normalized * 175)  # Range: 80-255
                    pen.setColor(QColor(100, 100, 100, alpha))

                    painter.setPen(pen)
                    painter.drawLine(line)
            else:
                # NORMAL MODE - uniform lines
                pen.setWidth(1)
                pen.setColor(QColor(100, 100, 100, 150))  # Consistent color
                painter.setPen(pen)
                lines_only = [x[0] for x in self.lines_data]
                painter.drawLines(lines_only)