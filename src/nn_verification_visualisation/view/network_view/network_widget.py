from typing import List, Callable

from PySide6.QtGui import QColor, QPainter, QPen, QWheelEvent
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QGraphicsView, QGraphicsLineItem, QGraphicsScene
from PySide6.QtCore import Qt

from nn_verification_visualisation.model.data.network_verification_config import NetworkVerificationConfig
from nn_verification_visualisation.view.network_view.network_edge_batch import NetworkEdgeBatch
from nn_verification_visualisation.view.network_view.network_node import NetworkNode


class NetworkWidget(QGraphicsView):
    configuration: NetworkVerificationConfig
    node_layers: List[List[NetworkNode]]
    scene: QGraphicsScene

    nodes_selectable: bool
    on_selection_changed: Callable[[List[NetworkNode]], None]

    background_color: QColor = QColor("#F0F0F0")
    min_size_x, min_size_y = (600, 400)
    padding_x, padding_y = (1000, 400)
    height_to_width_ration: float = 1.0
    node_spacing = 90
    radius = 18.0
    performance_mode_edge_threshold: int = 25000

    def __init__(self, configuration: NetworkVerificationConfig, nodes_selectable: bool = False, on_selection_changed: Callable[[List[NetworkNode]], None] = None):
        super().__init__()
        self.configuration = configuration

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.setRenderHint(QPainter.RenderHint.Antialiasing)

        self.node_layers = []

        self.setUpdatesEnabled(False)
        self.__build_network()
        self.setUpdatesEnabled(True)

        self.setBackgroundBrush(self.background_color)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.viewport().setMouseTracking(True)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setMinimumSize(self.min_size_x, self.min_size_y)
        self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.scene.setItemIndexMethod(QGraphicsScene.ItemIndexMethod.NoIndex)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)

        content_rect = self.scene.itemsBoundingRect()

        # Allows panning past the actual nodes.
        padded_rect = content_rect.adjusted(-self.padding_x, -self.padding_y, self.padding_x, self.padding_y)
        self.scene.setSceneRect(padded_rect)
        self.centerOn(content_rect.center())

    def __build_network(self):
        layers = self.configuration.layers_dimensions
        if not layers:
            return

        # compute bounding height by the tallest layer
        max_nodes = max(layers)
        total_height = max_nodes * self.node_spacing
        self.layer_spacing = total_height / (self.height_to_width_ration * (len(layers)-1))

        # add the nodes
        for i, num_nodes in enumerate(layers):
            print(f"Drawing Layer {i}, Number of Nodes: {num_nodes}")
            current_layer_nodes = []
            x = i * self.layer_spacing
            layer_height = num_nodes * self.node_spacing
            top_y = (total_height - layer_height) / 2

            for j in range(num_nodes):
                y = top_y + j * self.node_spacing
                node = NetworkNode(
                    index=j,
                    layer_index=i,
                    radius=self.radius,
                    selectable=False,
                    on_click=self._on_node_clicked
                )

                node.setPos(x, y)
                node.setZValue(1)
                self.scene.addItem(node)
                current_layer_nodes.append(node)

            self.node_layers.append(current_layer_nodes)

        total_edges = sum(layers[i] * layers[i + 1] for i in range(len(layers) - 1))
        self.use_performance_mode = total_edges > self.performance_mode_edge_threshold

        print(f"Total Edges: {total_edges}. Performance Mode: {self.use_performance_mode}")

        for i in range(len(self.node_layers) - 1):
            source_layer = self.node_layers[i]
            target_layer = self.node_layers[i + 1]

            # We pass the global decision into the item
            edge_item = NetworkEdgeBatch(
                source_layer,
                target_layer,
                force_block=self.use_performance_mode
            )
            self.scene.addItem(edge_item)

        self.scene.setSceneRect(self.scene.itemsBoundingRect())

    def selection_changed(self):
        if not self.on_selection_changed:
            return
        selected_nodes: List[NetworkNode] = [item for item in self.scene.selectedItems() if isinstance(item, NetworkNode)]
        self.on_selection_changed(selected_nodes)

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

        #if 0.7 < new_scale < 10.0:
        self.scale(zoom_factor, zoom_factor)

        event.accept()