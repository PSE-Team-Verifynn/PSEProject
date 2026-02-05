import random
from typing import List, Callable

import onnx
from PySide6.QtGui import QColor, QPainter, QPen, QWheelEvent, QKeyEvent, QTransform, QPalette, QBrush
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QSlider, QGraphicsLineItem, QGraphicsItem, QComboBox
from PySide6.QtCore import Qt, QVariantAnimation, QEasingCurve, QParallelAnimationGroup, QLineF
from onnx import numpy_helper, ModelProto

from nn_verification_visualisation.model.data.network_verification_config import NetworkVerificationConfig
from nn_verification_visualisation.view.network_view.network_edge_representation import NetworkEdgeBatch
from nn_verification_visualisation.view.network_view.network_node_representation import NetworkNode, NetworkLayerLine

from nn_verification_visualisation.view.dialogs.settings_dialog import SettingsDialog
from nn_verification_visualisation.view.dialogs.settings_option import SettingsOption


class NetworkWidget(QGraphicsView):
    configuration: NetworkVerificationConfig
    node_layers: List[List[NetworkNode]]
    scene: QGraphicsScene
    background_color: QColor

    nodes_selectable: bool
    on_selection_changed: Callable[[int, int], QColor | None]

    height_to_width_ration: float
    node_spacing = 90
    radius = 18.0

    # Thresholds
    performance_mode_edge_threshold: int = 25000  # Triggers block edges + LOD Nodes
    weighted_mode_edge_threshold: int = 500  # Triggers individual weighted edges
    use_performance_mode: bool
    use_weighted_mode: bool

    zoom_in_factor = 1.15
    zoom_out_factor = 1 / 1.15
    padding_percentage = 0.5

    remove_settings: List[Callable]

    def __init__(self, configuration: NetworkVerificationConfig, nodes_selectable: bool = False,
                 on_selection_changed: Callable[[int, int], QColor | None] = None):
        super().__init__()
        self.setFrameStyle(0)
        self.background_color = self.palette().color(QPalette.ColorRole.Base)
        self.configuration = configuration

        self.height_to_width_ration = 1.0

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

        self.remove_settings = []
        self.node_layers = []
        self.use_performance_mode = False
        self.use_weighted_mode = False
        self.manual_mode_override = False  # Track if user has manually selected a mode

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.setUpdatesEnabled(False)
        self.__build_network()
        self.setUpdatesEnabled(True)

        self.__update_view_constraints()
        self.fitInView(self.scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def get_height_to_width_changer(self):
        def height_to_width_changed(value: int):
            self.height_to_width_ration = value / 10
            self.node_layers = []
            self.scene.clear()
            self.setUpdatesEnabled(False)
            self.__build_network()
            self.setUpdatesEnabled(True)
            self.__update_view_constraints()
            self.fitInView(self.scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(5, 20)
        current_value = int(self.height_to_width_ration * 10)
        slider.setValue(current_value)
        slider.valueChanged.connect(height_to_width_changed)

        return slider

    def get_performance_mode_changer(self):
        def performance_mode_changed(value: int):
            self.manual_mode_override = True  # User has manually selected a mode
            self.use_performance_mode = value == 0
            self.use_weighted_mode = value == 1

            self.node_layers = []
            self.scene.clear()

            self.__build_network()

        dropdown = QComboBox()
        dropdown.addItems(["Performance Mode"])
        if not self.use_performance_mode or self.manual_mode_override:
            dropdown.addItems(["Weighted Mode", "Normal Mode"])
        dropdown.setCurrentIndex(0 if self.use_performance_mode else 1 if self.use_weighted_mode else 2)
        dropdown.currentIndexChanged.connect(performance_mode_changed)

        return dropdown

    def __build_network(self):
        layers = self.configuration.layers_dimensions
        if not layers:
            return

        total_edges = sum(layers[i] * layers[i + 1] for i in range(len(layers) - 1))

        # Auto-detect mode based on edge count, unless user has manually selected a mode
        if not self.manual_mode_override:
            # 1. Performance Mode (Large networks)
            if total_edges > self.performance_mode_edge_threshold:
                self.use_performance_mode = True
                self.use_weighted_mode = False
            # 2. Weighted Mode (Small networks)
            elif total_edges < self.weighted_mode_edge_threshold:
                self.use_performance_mode = False
                self.use_weighted_mode = True
            # 3. Normal Mode (Medium networks)
            else:
                self.use_performance_mode = False
                self.use_weighted_mode = False

        # Dimensions
        max_nodes = max(layers)
        total_height = max_nodes * self.node_spacing
        self.layer_spacing = total_height / (self.height_to_width_ration * (len(layers) - 1))

        # --- Build Nodes & Layers ---
        for i, num_nodes in enumerate(layers):
            current_layer_nodes = []
            x = i * self.layer_spacing
            layer_height = num_nodes * self.node_spacing
            top_y = (total_height - layer_height) / 2

            # Visual Proxy for LOD Mode (The "Line" representation)
            if self.use_performance_mode:
                # Draw a line from the top node position to bottom node position
                bot_y = top_y + (num_nodes - 1) * self.node_spacing
                layer_line = NetworkLayerLine(x, top_y, bot_y)
                self.scene.addItem(layer_line)

            for j in range(num_nodes):
                y = top_y + j * self.node_spacing
                node = NetworkNode(
                    index=j,
                    layer_index=i,
                    radius=self.radius,
                    selectable=False,
                    on_click=self._on_node_clicked
                )

                if self.use_performance_mode:
                    node.set_lod_mode(True)  # Nodes will hide themselves when zoomed out

                node.setPos(x, y)
                node.setZValue(1)
                self.scene.addItem(node)
                current_layer_nodes.append(node)

            self.node_layers.append(current_layer_nodes)

        all_weights = None
        if self.use_weighted_mode:
            all_weights = self.get_weights_from_onnx(self.configuration.network.model)

        # --- Build Edges ---
        for i in range(len(self.node_layers) - 1):
            source_layer = self.node_layers[i]
            target_layer = self.node_layers[i + 1]

            layer_weights = None
            if self.use_weighted_mode and all_weights:
                # Pass ONLY the matrix for this specific connection
                layer_weights = all_weights[i]

            edge_item = NetworkEdgeBatch(
                source_layer,
                target_layer,
                force_block=self.use_performance_mode,
                use_weighted=self.use_weighted_mode,
                weights=layer_weights
            )
            self.scene.addItem(edge_item)

        self.scene.setSceneRect(self.scene.itemsBoundingRect())

    # ... (Rest of methods: _on_node_clicked, __update_view_constraints, go_to_node, etc. remain unchanged) ...
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

        # Target Zoom Level
        target_scale = 60.0 / self.radius
        current_scale = self.transform().m11()

        self.zoom_anim = QVariantAnimation()
        self.zoom_anim.setDuration(600)
        self.zoom_anim.setStartValue(current_scale)
        self.zoom_anim.setEndValue(target_scale)
        self.zoom_anim.setEasingCurve(QEasingCurve.Type.InOutCubic)

        def apply_zoom(scale):
            self.setTransform(QTransform.fromScale(scale, scale))
            self.centerOn(target_scene_pos)

        self.zoom_anim.valueChanged.connect(apply_zoom)

        self.anim_group = QParallelAnimationGroup()
        self.anim_group.addAnimation(self.zoom_anim)
        self.anim_group.finished.connect(lambda: node.setSelected(True))

        self.centerOn(target_scene_pos)
        self.anim_group.start()

    def unselect_node(self, layer_index: int, node_index: int):
        self.node_layers[layer_index][node_index].setBrush(NetworkNode.color_unselected)

    def select_node(self, layer_index: int, node_index: int, color: QColor):
        self.node_layers[layer_index][node_index].setBrush(color)

    def get_weights_from_onnx(self, model_proto: ModelProto):
        # 1. Create a lookup table for all initializers (weights/biases)
        # This maps the name of the tensor to its actual numerical array
        weights_dict = {}
        for initializer in model_proto.graph.initializer:
            weights_dict[initializer.name] = numpy_helper.to_array(initializer)

        # 2. Map nodes to their weights
        # In a typical Fully Connected (Gemm) layer:
        # node.input[0] is the input data
        # node.input[1] is the weight matrix
        # node.input[2] is the bias (optional)
        layers_weights = []

        for node in model_proto.graph.node:
            if node.op_type in ["Gemm", "MatMul"]:
                weight_name = node.input[1]
                if weight_name in weights_dict:
                    layers_weights.append(weights_dict[weight_name])

        return layers_weights

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_R:
            layer = random.randint(0, len(self.node_layers) - 1)
            node = random.randint(0, len(self.node_layers[layer]) - 1)
            self.go_to_node(layer, node)
        else:
            super().keyPressEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        zoom_factor = self.zoom_in_factor if event.angleDelta().y() > 0 else self.zoom_out_factor
        current_scale = self.transform().m11()
        new_scale = current_scale * zoom_factor

        if self.min_scale < new_scale < self.max_scale:
            self.scale(zoom_factor, zoom_factor)

        event.accept()

    def hideEvent(self, event):
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.NoViewportUpdate)
        self.scene.setItemIndexMethod(QGraphicsScene.ItemIndexMethod.NoIndex)
        self.setUpdatesEnabled(False)

        if self.remove_settings:
            for remove_setting in self.remove_settings:
                remove_setting()
            self.remove_settings.clear() # Prevent multiple calls
        super().hideEvent(event)

    def showEvent(self, event):
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)
        self.setUpdatesEnabled(True)

        if not self.nodes_selectable:
            self.remove_settings.append(SettingsDialog.add_setting(
                SettingsOption("Network Height To Width Ratio", self.get_height_to_width_changer,
                               f"Network Display: {self.configuration.network.name}")))
            self.remove_settings.append(SettingsDialog.add_setting(
                SettingsOption("Performance Mode", self.get_performance_mode_changer,
                               f"Network Display: {self.configuration.network.name}")))
        super().showEvent(event)
