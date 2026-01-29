from typing import List, Callable, Tuple
from numpy import clip

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QWidget, QSplitter, QLabel, QHBoxLayout,
                               QVBoxLayout, QComboBox, QSpinBox, QPushButton, QLayout)

# Assuming these imports exist in your project structure
from nn_verification_visualisation.model.data.plot_generation_config import PlotGenerationConfig
from nn_verification_visualisation.model.data.storage import Storage
from nn_verification_visualisation.view.dialogs.dialog_base import DialogBase
from nn_verification_visualisation.view.network_view.network_widget import NetworkWidget


class NeuronPicker(DialogBase):
    # Type hinting references (initialized in __init__ to prevent state persistence issues)
    current_network: int
    current_algorithm: str
    current_neurons: List[Tuple[int, int]]
    parameters: List[str]
    num_neurons: int
    max_neuron_num_per_layer: List[int]

    network_widget: NetworkWidget = None
    node_spin_boxes: List[Tuple[QSpinBox, QSpinBox]]
    network_presentation: QVBoxLayout
    algorithm_selector: QComboBox

    def __init__(self, on_close: Callable[[], None], num_neurons: int = 2):
        self.num_neurons = num_neurons
        self.current_network = 0
        self.current_algorithm = ""
        self.current_neurons = [(0, 0) for _ in range(num_neurons)]
        self.node_spin_boxes = []
        self.max_neuron_num_per_layer = []
        self.parameters = []

        self.algorithm_selector = QComboBox()
        self.algorithm_selector.currentIndexChanged.connect(
            self.__on_change_algorithm
        )

        # Update the algorithm list on change
        Storage().algorithm_change_listeners.append(self.update_algorithms)

        super().__init__(on_close, "Neuron Picker", has_title=False)

    def update_algorithms(self):
        index = self.algorithm_selector.currentIndex() if self.algorithm_selector.currentIndex() > -1 else 0
        self.algorithm_selector.blockSignals(True)

        self.algorithm_selector.clear()

        for algorithm in Storage().algorithms:
            self.algorithm_selector.addItem(algorithm.name)

        self.algorithm_selector.setCurrentIndex(index)

        self.algorithm_selector.blockSignals(False)

    def get_content(self) -> QWidget:
        network_picker_layout = QVBoxLayout()
        network_picker_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setContentsMargins(0, 0, 0, 0)
        splitter.setChildrenCollapsible(False)

        # 1. Setup Side Bar (This creates the spinboxes)
        side_bar = QWidget()
        side_bar.setObjectName("dialog-sidebar")
        side_bar.setMinimumWidth(250)
        side_bar.setLayout(self.__get_side_bar_content())

        # 2. Setup Network Presentation
        self.network_presentation = QVBoxLayout()
        network_widget_container = QWidget()
        network_widget_container.setLayout(self.network_presentation)
        network_widget_container.setMinimumWidth(500)

        # 3. Initialize Data (Now that UI exists)
        if len(Storage().networks) == 0:
            self.network_presentation.addWidget(QLabel("No networks loaded"))
        else:
            self.__on_change_network(0)

        if len(Storage().algorithms) == 0:
            self.current_algorithm = ""
        else:
            self.current_algorithm = Storage().algorithms[0].name

        self.network_presentation.addLayout(self.__get_button_row()) # Buttons

        splitter.setMinimumHeight(500)
        splitter.addWidget(side_bar)
        splitter.addWidget(network_widget_container)
        splitter.setObjectName("transparent")

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)

        network_picker_layout.addWidget(splitter)

        widget = QWidget()
        widget.setLayout(network_picker_layout)

        return widget

    def __get_button_row(self) -> QLayout:
        move_buttons = QHBoxLayout()
        back_button = QPushButton("Back")
        back_button.clicked.connect(self.close)
        continue_button = QPushButton("Continue")
        continue_button.clicked.connect(self.on_close)

        move_buttons.addWidget(back_button)
        move_buttons.addWidget(continue_button)
        move_buttons.setAlignment(Qt.AlignmentFlag.AlignRight)
        return move_buttons

    def construct_config(self) -> PlotGenerationConfig:
        if len(Storage().networks) < self.current_network - 1 or len(
                Storage().networks) == 0 or self.current_algorithm == "":
            return None
        network = Storage().networks[self.current_network]
        matching_algorithms = [alg for alg in Storage().algorithms if alg.name == self.current_algorithm]
        if not matching_algorithms:
            print("Wrong algorithm selected!")
            return None
        algorithm = matching_algorithms[0]
        return PlotGenerationConfig(network, algorithm, self.current_neurons, [])

    def __on_change_network(self, index: int):
        if not Storage().networks:
            return

        print(f"Network changed to {Storage().networks[index].network.name}")
        new_network = Storage().networks[index]
        self.current_network = index

        # Re-create the network widget
        self.network_widget = NetworkWidget(Storage().networks[index])

        # Update limits based on new network
        self.max_neuron_num_per_layer = new_network.layers_dimensions

        # Update Layer Spinbox Ranges
        layer_count = len(new_network.layers_dimensions)
        for (layer_spin, neuron_spin) in self.node_spin_boxes:
            # Block signals to prevent triggering logic while setting up
            layer_spin.blockSignals(True)
            layer_spin.setRange(0, layer_count - 1)
            layer_spin.setValue(0)
            layer_spin.blockSignals(False)

            # Reset neuron spin based on layer 0
            neuron_spin.blockSignals(True)
            if layer_count > 0:
                max_nodes = self.max_neuron_num_per_layer[0]
                neuron_spin.setRange(0, max_nodes - 1 if max_nodes > 0 else 0)
            neuron_spin.setValue(0)
            neuron_spin.blockSignals(False)

        # Reset stored choices
        self.current_neurons = [(0, 0) for _ in range(self.num_neurons)]

        # Visual update
        if self.network_presentation.count() > 0:
            # Safely remove existing widget
            item = self.network_presentation.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.network_presentation.addWidget(self.network_widget)

    def __on_change_algorithm(self, index: int):
        if Storage().algorithms:
            self.current_algorithm = Storage().algorithms[index].name

    def __on_change_layer_choice(self, selection_index: int, layer_index: int):
        # Safety check for index out of bounds
        if layer_index >= len(self.max_neuron_num_per_layer):
            return

        new_max_node_num = self.max_neuron_num_per_layer[layer_index]

        new_node_index = clip(self.current_neurons[selection_index][1], 0, new_max_node_num - 1)
        # Update the stored tuple
        self.current_neurons[selection_index] = (layer_index, new_node_index)

        # Update UI constraints
        spin_box = self.node_spin_boxes[selection_index][1]
        spin_box.blockSignals(True)  # Prevent recursion
        spin_box.setRange(0, new_max_node_num - 1 if new_max_node_num > 0 else 0)
        spin_box.setValue(new_node_index)
        spin_box.blockSignals(False)

    def __on_change_choice_within_layer(self, selection_index: int, node_index: int):
        current_layer = self.current_neurons[selection_index][0]
        self.current_neurons[selection_index] = (current_layer, node_index)

    def __jump_to_neuron(self, neuron_index: int):
        if self.network_widget:
            layer, node = self.current_neurons[neuron_index]
            self.network_widget.go_to_node(layer, node)

    def __get_side_bar_content(self) -> QVBoxLayout:
        layout = QVBoxLayout()

        title = QLabel("Neuron Picker")
        title.setObjectName("title")

        layout.addWidget(title)

        # --- Network Selector ---
        network_group = QHBoxLayout()
        network_group.addWidget(QLabel("Network:"))
        network_selector = QComboBox()
        for network in Storage().networks:
            network_selector.addItem(network.network.name)
        network_selector.currentIndexChanged.connect(self.__on_change_network)
        network_group.addWidget(network_selector)

        # --- Algorithm Selector ---
        algorithm_group = QHBoxLayout()
        algorithm_group.addWidget(QLabel("Algorithm:"))
        self.update_algorithms()
        algorithm_group.addWidget(self.algorithm_selector)

        layout.addLayout(network_group)
        layout.addLayout(algorithm_group)

        # --- Neuron Pair Selectors ---
        for i in range(0, self.num_neurons):
            neuron_pair_group = QHBoxLayout()
            label = QLabel(f"Pair {i + 1}")
            label.setAlignment(Qt.AlignmentFlag.AlignBottom)
            neuron_pair_group.addWidget(label)

            # Color Circle
            color = "#35C54B"  # TODO: Implement automatic color selection
            color_circle = QLabel()
            color_circle.setFixedSize(16, 16)
            color_circle.setStyleSheet(f"background-color: {color}; border-radius: 8px;")
            neuron_pair_group.addWidget(color_circle, alignment=Qt.AlignmentFlag.AlignBottom)

            # Controls
            layer_spinbox = QSpinBox()
            layer_spinbox.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
            layer_spinbox.setFixedWidth(48)
            layer_spinbox.setAlignment(Qt.AlignmentFlag.AlignLeft)

            neuron_spinbox = QSpinBox()
            neuron_spinbox.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
            neuron_spinbox.setFixedWidth(48)
            neuron_spinbox.setAlignment(Qt.AlignmentFlag.AlignRight)

            self.node_spin_boxes.append((layer_spinbox, neuron_spinbox))

            layer_spinbox.valueChanged.connect(
                lambda new_val, idx=i: self.__on_change_layer_choice(idx, new_val)
            )
            neuron_spinbox.valueChanged.connect(
                lambda new_val, idx=i: self.__on_change_choice_within_layer(idx, new_val)
            )

            colon = QLabel(":")
            colon.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignBottom)
            colon.setFixedWidth(12)

            layer_hint = QLabel("Layer")
            node_hint = QLabel("Node")
            node_hint.setAlignment(Qt.AlignmentFlag.AlignRight)

            eye_button = QPushButton("👀")
            eye_button.setObjectName("transparent-button")
            eye_button.clicked.connect(lambda _, idx=i: self.__jump_to_neuron(idx))

            layer_box = QVBoxLayout()
            layer_box.addWidget(layer_hint)
            layer_box.addWidget(layer_spinbox)
            layer_box.setAlignment(Qt.AlignmentFlag.AlignBottom)

            node_box = QVBoxLayout()
            node_box.addWidget(node_hint)
            node_box.addWidget(neuron_spinbox)
            node_box.setAlignment(Qt.AlignmentFlag.AlignBottom)

            neuron_pair_group.addStretch()
            neuron_pair_group.addLayout(layer_box)
            neuron_pair_group.addWidget(colon)
            neuron_pair_group.addLayout(node_box)
            neuron_pair_group.addWidget(eye_button)

            layout.addLayout(neuron_pair_group)

        layout.addStretch()

        return layout
