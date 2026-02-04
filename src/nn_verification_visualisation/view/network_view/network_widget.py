import random
from typing import List, Callable

from PySide6.QtGui import QColor, QPainter, QPen, QWheelEvent, QKeyEvent, QTransform, QPalette
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QSlider
from PySide6.QtCore import Qt, QVariantAnimation, QEasingCurve, QParallelAnimationGroup

from nn_verification_visualisation.model.data.network_verification_config import NetworkVerificationConfig
from nn_verification_visualisation.view.network_view.network_edge_batch import NetworkEdgeBatch
from nn_verification_visualisation.view.network_view.network_node import NetworkNode

from nn_verification_visualisation.view.dialogs.settings_dialog import SettingsDialog
from nn_verification_visualisation.view.dialogs.settings_option import SettingsOption


class NetworkWidget(QGraphicsView):

    configuration: NetworkVerificationConfig
    node_layers: List[List[NetworkNode]]
    scene: QGraphicsScene
    background_color: QColor

    nodes_selectable: bool
    on_selection_changed: Callable[[int, int], QColor | None]

    height_to_width_ration: float = 1.0
    node_spacing = 90
    radius = 18.0
    performance_mode_edge_threshold: int = 25000
    zoom_in_factor = 1.15
    zoom_out_factor = 1 / 1.15
    padding_percentage = 0.5

    def __init__(self, configuration: NetworkVerificationConfig, nodes_selectable: bool = False, on_selection_changed: Callable[[int, int], QColor | None] = None):
        super().__init__()
        self.setFrameStyle(0)
        self.background_color = self.palette().color(QPalette.ColorRole.Base)
        self.configuration = configuration

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.nodes_selectable = nodes_selectable
        self.on_selection_changed = on_selection_changed

        self.setBackgroundBrush(self.background_color)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.viewport().setMouseTracking(True)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.scene.setItemIndexMethod(QGraphicsScene.ItemIndexMethod.NoIndex)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.node_layers = []

        self.setUpdatesEnabled(False)
        self.__build_network()
        self.setUpdatesEnabled(True)

        self.__update_view_constraints()

        # Initial View
        self.fitInView(self.scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)

        SettingsDialog.add_setting(SettingsOption("Network Height To Width Ratio", self.get_height_to_width_changer, f"Network Display: {self.configuration.network.name}"))

    def get_height_to_width_changer(self):
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(5, 20)
        current_value = int(self.height_to_width_ration * 10)
        slider.setValue(current_value)
        slider.valueChanged.connect(self.height_to_width_changed)

        return slider

    def height_to_width_changed(self, value: int):
        self.height_to_width_ration = value / 10
        self.node_layers = []
        self.scene.clear()
        self.__build_network()


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

    def _on_node_clicked(self, position: tuple[int, int]) -> None:
        if not self.nodes_selectable:
            return
        layer, index = position
        print(f"Clicked Node: Layer {layer}, Index {index}")
        if self.on_selection_changed:
            new_color = self.on_selection_changed(layer, index)
            if new_color is None:
                return
            self.node_layers[layer][index].setBrush(new_color)

    def __update_view_constraints(self):
        content_rect = self.scene.itemsBoundingRect()

        if content_rect.isEmpty():
            return

        self.padding_x = content_rect.width() * self.padding_percentage
        self.padding_y = content_rect.height() * self.padding_percentage

        padded_rect = content_rect.adjusted(
            -self.padding_x, -self.padding_y,
            self.padding_x, self.padding_y
        )
        self.scene.setSceneRect(padded_rect)

        viewport_width = self.viewport().width() or 800
        self.min_scale = (viewport_width / padded_rect.width()) * 0.8
        self.max_scale = 300 / self.radius

    def go_to_node(self, layer_index: int, node_index: int):
        if not self.node_layers or not self.node_layers[layer_index]:
            return

        node = self.node_layers[layer_index][node_index]
        target_scene_pos = node.scenePos()

        # 1. Determine Target Zoom Level
        # We want the node to occupy a specific size on screen (e.g., 60px wide)
        # Target Scale = Desired Pixel Size / Actual Item Radius
        target_scale = 60.0 / self.radius
        current_scale = self.transform().m11()

        self.zoom_anim = QVariantAnimation()
        self.zoom_anim.setDuration(600)
        self.zoom_anim.setStartValue(current_scale)
        self.zoom_anim.setEndValue(target_scale)
        self.zoom_anim.setEasingCurve(QEasingCurve.Type.InOutCubic)

        def apply_zoom(scale):
            # Apply the new scale while maintaining the current transform's center
            self.setTransform(QTransform.fromScale(scale, scale))
            self.centerOn(target_scene_pos)


        self.zoom_anim.valueChanged.connect(apply_zoom)

        # 5. Run both together
        self.anim_group = QParallelAnimationGroup()
        self.anim_group.addAnimation(self.zoom_anim)

        # Visual feedback: Highlight the node when reached
        self.anim_group.finished.connect(lambda: node.setSelected(True))

        self.centerOn(target_scene_pos)
        self.anim_group.start()

    def unselect_node(self, layer_index: int, node_index: int):
        self.node_layers[layer_index][node_index].setBrush(NetworkNode.color_unselected)

    def select_node(self, layer_index: int, node_index: int, color: QColor):
        self.node_layers[layer_index][node_index].setBrush(color)

    def keyPressEvent(self, event: QKeyEvent):
        """
        Captured every time a key is pressed while the widget has focus.
        """
        # Example: Press 'R' to go to a random node
        if event.key() == Qt.Key.Key_R:
            print("R key pressed: Jumping to random node.")
            layer = random.randint(0, len(self.node_layers) - 1)
            node = random.randint(0, len(self.node_layers[layer]) - 1)
            self.go_to_node(layer, node)
        else:
            super().keyPressEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        zoom_factor = self.zoom_in_factor if event.angleDelta().y() > 0 else self.zoom_out_factor

        # Check current scale (m11 is the horizontal scaling factor)
        current_scale = self.transform().m11()
        new_scale = current_scale * zoom_factor

        # Apply limits calculated in __update_view_constraints
        if self.min_scale < new_scale < self.max_scale:
            self.scale(zoom_factor, zoom_factor)

        event.accept()