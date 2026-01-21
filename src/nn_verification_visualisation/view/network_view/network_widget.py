from typing import List

from PySide6.QtGui import QColor, QPainter, QPen, QWheelEvent
from PySide6.QtWidgets import QGraphicsView, QGraphicsLineItem, QGraphicsScene
from PySide6.QtCore import Qt

from nn_verification_visualisation.model.data.network_verification_config import NetworkVerificationConfig
from nn_verification_visualisation.view.network_view.network_node import NetworkNode


class NetworkWidget(QGraphicsView):
    configuration: NetworkVerificationConfig
    node_layers: List[List[NetworkNode]]
    edges: List[QGraphicsLineItem]
    scene: QGraphicsScene

    def __init__(self, configuration: NetworkVerificationConfig):
        super().__init__()
        self.configuration = configuration

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.setRenderHint(QPainter.RenderHint.Antialiasing)

        self.node_layers = []
        self.edges = []

        self.__build_network()

        self.setBackgroundBrush(QColor("#F0F0F0"))
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.viewport().setMouseTracking(True)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)
        self.setMinimumSize(800, 600)
        self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

        # Hide scrollbars for a cleaner look (navigation via zoom/pan)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content_rect = self.scene.itemsBoundingRect()

        # 2. Define a large margin (e.g., 1000 pixels or based on view size)
        # This allows you to pan far past the actual nodes.
        padding = 500
        padded_rect = content_rect.adjusted(-padding, -padding, padding, padding)

        # 3. Set the scene rect to this larger area
        self.scene.setSceneRect(padded_rect)

        # 4. Initially center the view on the actual content
        self.centerOn(content_rect.center())

    def __build_network(self):
        layers = self.configuration.layers_dimensions
        if not layers:
            return

        # layout parameters
        layer_spacing = 220
        node_spacing = 90
        margin = 40
        radius = 18.0

        # compute bounding height by the tallest layer
        max_nodes = max(layers)
        total_height = max_nodes * node_spacing + 2*margin

        # add the nodes
        for i, num_nodes in enumerate(layers):
            print(f"Layer {i}, Number of Nodes: {num_nodes}")
            current_layer_nodes = []
            x = i * layer_spacing
            layer_height = num_nodes * node_spacing
            top_y = (total_height - layer_height) / 2

            for j in range(num_nodes):
                print(f"Adding Node {j}")
                y = top_y + j * node_spacing
                # Important: avoid lambda capturing loop variables. use partial to freeze args.
                node = NetworkNode(
                    index=j,
                    layer_index=i,
                    radius=18.0,
                    selectable=False,
                    on_click=self._on_node_clicked
                )

                node.setPos(x, y)
                node.setZValue(1)
                self.scene.addItem(node)
                current_layer_nodes.append(node)

            self.node_layers.append(current_layer_nodes)

        # connect the layers by adding edges
        for i in range(len(self.node_layers) - 1):
            source_layer = self.node_layers[i]
            target_layer = self.node_layers[i + 1]

            if len(source_layer) > 20 or len(target_layer) > 20:
                print(f"Sorry no edges since layer is {max(len(source_layer), len(target_layer))} big")
                continue

            for source_node in source_layer:
                for target_node in target_layer:

                    print(f"Adding Edge from {source_node.layer_index}, {source_node.index} to {target_node.index}")
                    line = QGraphicsLineItem(
                        source_node.scenePos().x(), source_node.scenePos().y(),
                        target_node.scenePos().x(), target_node.scenePos().y()
                    )

                    # Style the line
                    pen = QPen(QColor("gray"))
                    pen.setWidth(1)
                    line.setPen(pen)
                    line.setZValue(-1)

                    self.scene.addItem(line)
                    self.edges.append(line)

        self.scene.setSceneRect(self.scene.itemsBoundingRect())


    def __on_selection_changed(self):
        pass

    def _on_node_clicked(self, position: tuple[int, int]) -> None:
        layer, index = position
        print(f"Clicked Node: Layer {layer}, Index {index}")
        # You can emit a signal here or update a side panel

    def wheelEvent(self, event: QWheelEvent):
        """
        Handles mouse wheel scrolling to zoom in and out.
        """
        # 1. Define zoom settings
        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor

        # 2. Set anchor so we zoom relative to the mouse cursor position
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        # 3. Determine zoom direction
        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor

        # 4. Apply zoom with safety limits
        # Check current total scale by looking at the transformation matrix
        current_scale = self.transform().m11()
        new_scale = current_scale * zoom_factor

        if 0.7 < new_scale < 10.0:
            self.scale(zoom_factor, zoom_factor)

        event.accept()