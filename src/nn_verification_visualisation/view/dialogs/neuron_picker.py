import random
from typing import List, Callable, Tuple

from PySide6.QtGui import QColor, QIcon

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QWidget, QSplitter, QLabel, QHBoxLayout,
                               QVBoxLayout, QComboBox, QSpinBox, QPushButton, QLayout)

# Assuming these imports exist in your project structure
from nn_verification_visualisation.model.data.plot_generation_config import PlotGenerationConfig
from nn_verification_visualisation.model.data.storage import Storage
from nn_verification_visualisation.utils.result import Result, Failure, Success
from nn_verification_visualisation.view.dialogs.dialog_base import DialogBase
from nn_verification_visualisation.view.dialogs.run_samples_dialog import RunSamplesDialog
from nn_verification_visualisation.view.base_view.sample_metrics import SampleMetricsWidget
from nn_verification_visualisation.view.dialogs.sample_results_dialog import SampleResultsDialog
from nn_verification_visualisation.view.base_view.bounds_display import BoundsDisplayWidget
from nn_verification_visualisation.view.network_view.network_widget import NetworkWidget


def get_neuron_colors(num_neurons) -> List[QColor]:
    """
        Returns a list of n hex codes for neuron colors.
        Based on the Okabe-Ito palette (optimized for color blindness).
        If more than 8 neurons are selected, random colors are used to fill up the list.
    """
    # The Okabe-Ito palette (optimized for color blindness)
    # Black is left out for contrast reasons
    palette = [
        QColor("#E69F00"), QColor("#56B4E9"), QColor("#009E73"), QColor("#F0E442"),
        QColor("#0072B2"), QColor("#D55E00"), QColor("#CC79A7")
    ]

    if num_neurons <= len(palette):
        return palette[:num_neurons]

    # Generates predictable random colors
    rng = random.Random(0)
    extra_colors = []
    for _ in range(num_neurons - len(palette)):
        extra_colors.append(QColor("#%06x" % rng.randint(0, 0xFFFFFF)))

    return palette + extra_colors


class NeuronPicker(DialogBase):
    """
    Manages selection of the configuration that can then be used to generate plots.

    This class is responsible for creating and managing a dialog where users can
    select neurons from a neural network, configure algorithms, and choose input bounds
    for further processing.
    """
    current_network: int
    current_algorithm: str
    current_neurons: List[Tuple[int, int]]
    parameters: List[str]
    num_neurons: int
    max_neuron_num_per_layer: List[int]

    neuron_selection_index: int
    neuron_colors: List[QColor]

    network_widget: NetworkWidget | None
    node_spin_boxes: List[Tuple[QSpinBox, QSpinBox]]
    network_presentation: QVBoxLayout
    algorithm_selector: QComboBox
    network_selector: QComboBox
    bounds_selector: QComboBox | None
    max_bounds_display_inputs: int

    def __init__(self, on_close: Callable[[], None], num_neurons: int = 2, preset: PlotGenerationConfig | None = None):
        self.current_network = 0
        self.current_algorithm = ""

        if preset is not None:
            self.num_neurons = len(preset.selected_neurons)
        else:
            self.num_neurons = num_neurons

        self.current_neurons = [(0, 0) for _ in range(self.num_neurons)]

        self.network_widget = None
        self.node_spin_boxes = []
        self.max_neuron_num_per_layer = []
        self.parameters = []
        self.neuron_selection_index = 0
        self.neuron_colors = get_neuron_colors(self.num_neurons)

        self.network_selector = QComboBox()
        for network in Storage().networks:
            self.network_selector.addItem(network.network.name)
        self.network_selector.currentIndexChanged.connect(self.__on_change_network)

        self.algorithm_selector = QComboBox()
        self.algorithm_selector.currentIndexChanged.connect(
            self.__on_change_algorithm
        )
        self.bounds_selector = None
        self.bounds_display_group = None
        self.sample_metrics = None
        self._bounds_index_label_width = 36

        # Update the algorithm list on change
        Storage().algorithm_change_listeners.append(self.update_algorithms)

        super().__init__(on_close, "Neuron Picker", has_title=False)

        if preset is not None:
            self.__load_from_config(preset)
            
    def __compute_bounds_index_label_width(self, input_count: int) -> int:
        if self.bounds_display_group is None:
            return 36
        if input_count <= 0:
            return 36
        max_index = input_count - 1
        sample_text = f"{max_index}:"
        metrics = self.bounds_display_group.fontMetrics()
        text_width = metrics.horizontalAdvance(sample_text)
        # Add a small padding buffer so 3+ digits don't clip.
        return max(36, text_width + 8)

    def update_algorithms(self) -> None:
        """
        Refreshes the algorithm selector.
        This is used to provide an up-to-date list of algorithms to the user.
        """
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

        side_bar = QWidget()
        side_bar.setObjectName("dialog-sidebar")
        side_bar.setMinimumWidth(270)
        side_bar.setLayout(self.__get_side_bar_content())

        self.network_presentation = QVBoxLayout()
        network_widget_container = QWidget()
        network_widget_container.setLayout(self.network_presentation)
        network_widget_container.setMinimumWidth(500)

        if len(Storage().networks) == 0:
            self.network_presentation.addWidget(QLabel("No networks loaded"))
        else:
            self.__on_change_network(self.current_network)

        if len(Storage().algorithms) != 0 and self.current_algorithm == "":
            self.current_algorithm = Storage().algorithms[0].name

        self.network_presentation.addLayout(self.__get_button_row())  # Buttons

        splitter.setMinimumHeight(500)
        splitter.addWidget(side_bar)
        splitter.addWidget(network_widget_container)
        splitter.setObjectName("transparent")

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        network_picker_layout.addWidget(splitter)

        widget = QWidget()
        widget.setLayout(network_picker_layout)

        return widget

    def __get_button_row(self) -> QLayout:
        move_buttons = QHBoxLayout()
        run_samples_button = QPushButton("Run Samples")
        run_samples_button.clicked.connect(self.__on_run_samples_clicked)
        back_button = QPushButton("Back")
        back_button.clicked.connect(self.close)
        continue_button = QPushButton("Continue")
        continue_button.clicked.connect(self.on_close)

        move_buttons.addWidget(run_samples_button)
        move_buttons.addWidget(back_button)
        move_buttons.addWidget(continue_button)
        move_buttons.setAlignment(Qt.AlignmentFlag.AlignRight)
        return move_buttons

    def construct_config(self) -> Result[PlotGenerationConfig]:
        """
        Uses the current state of the UI to construct a PlotGenerationConfig.
        This config represents the user choice on how to generate the plot.
        :return: a result of the constructed PlotGenerationConfig.
        """
        if len(Storage().networks) < self.current_network - 1 or len(
                Storage().networks) == 0 or self.current_algorithm == "":
            return Failure(Exception("No network selected - please load a network first"))
        network = Storage().networks[self.current_network]
        matching_algorithms = [alg for alg in Storage().algorithms if alg.name == self.current_algorithm]
        if not matching_algorithms:
            return Failure(Exception("No algorithm selected - please load an algorithm first"))

        algorithm = matching_algorithms[0]

        if self.bounds_selector is None or self.bounds_selector.currentIndex() < 0:
            return Failure(Exception("No bounds selected - please load bounds for the chosen network first"))

        bounds_index = self.bounds_selector.currentIndex()

        return Success(PlotGenerationConfig(network, algorithm, self.current_neurons, [], bounds_index=bounds_index))

    def __load_from_config(self, config: PlotGenerationConfig):
        def __sync_ui_to_internal_state():
            """
            Helper to update SpinBoxes and NetworkWidget based on self.current_neurons
            without triggering recursive logic signals.
            """
            if not self.network_widget:
                return

            # Clear default selections made by __on_change_network
            # We do this blindly because we are about to re-select everything
            for i in range(len(self.max_neuron_num_per_layer)):
                max_n = self.max_neuron_num_per_layer[i]
                for n in range(max_n):
                    self.network_widget.unselect_node(i, n)

            for i, (i_layer, i_neuron) in enumerate(self.current_neurons):
                if i >= len(self.node_spin_boxes):
                    break

                layer_spin, neuron_spin = self.node_spin_boxes[i]

                # Block signals to prevent "valueChanged" from triggering logic
                layer_spin.blockSignals(True)
                neuron_spin.blockSignals(True)

                # 1. Set Layer
                layer_spin.setValue(i_layer)

                # 2. Update Neuron Spinbox Range based on that layer
                # (We must do this manually because we blocked the signal that usually does it)
                new_max_node_num = self.max_neuron_num_per_layer[i_layer]
                neuron_spin.setRange(0, new_max_node_num - 1 if new_max_node_num > 0 else 0)

                # 3. Set Neuron
                neuron_spin.setValue(i_neuron)

                # Unblock
                layer_spin.blockSignals(False)
                neuron_spin.blockSignals(False)

                # 4. Update Visual Graph
                self.network_widget.select_node(i_layer, i_neuron, self.neuron_colors[i])

        nn_path_preset = config.nnconfig.network.path
        nn_indices = [i for i in range(len(Storage().networks)) if Storage().networks[i].network.path == nn_path_preset]
        nn_index = 0 if not nn_indices else nn_indices[0]

        algorithm_path_preset = config.algorithm.path
        algo_indices = [i for i, alg in enumerate(Storage().algorithms) if alg.path == algorithm_path_preset]
        algo_index = algo_indices[0] if algo_indices else 0

        self.network_selector.blockSignals(True)
        self.network_selector.setCurrentIndex(nn_index)
        self.network_selector.blockSignals(False)
        self.__on_change_network(nn_index)

        self.algorithm_selector.blockSignals(True)
        self.algorithm_selector.setCurrentIndex(algo_index)
        self.algorithm_selector.blockSignals(False)
        self.__on_change_algorithm(algo_index)

        self.current_neurons = []
        for i, (layer, neuron) in enumerate(config.selected_neurons):
            if layer < len(self.max_neuron_num_per_layer):
                max_nodes = self.max_neuron_num_per_layer[layer]
                valid_neuron = min(neuron, max_nodes - 1) if max_nodes > 0 else 0
                self.current_neurons.append((layer, valid_neuron))
            else:
                self.current_neurons.append((0, 0))

        print("Loading bounds: " + str(config.bounds_index))
        if self.bounds_selector and 0 <= config.bounds_index < self.bounds_selector.count():
            self.bounds_selector.setCurrentIndex(config.bounds_index)

        __sync_ui_to_internal_state()

    def __on_change_network(self, index: int):
        if not Storage().networks:
            return

        print(f"Network changed to {Storage().networks[index].network.name}")
        new_network = Storage().networks[index]
        self.current_network = index
        if self.bounds_selector is not None:
            self.__populate_bounds_selector(index)
        self.__rebuild_bounds_display_rows()
        self.__update_bounds_display()
        if hasattr(self, "bounds_toggle_button") and self.bounds_toggle_button is not None:
            self.bounds_toggle_button.setVisible(True)
        self.__update_sample_results()

        self.network_widget = NetworkWidget(Storage().networks[index], nodes_selectable=True,
                                            on_selection_changed=self.__on_node_selection_change)

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
        self.current_neurons = [(0, min(i, self.max_neuron_num_per_layer[i] -1)) for i in range(self.num_neurons)]

        # Visual update
        if self.network_presentation.count() > 0:
            # Safely remove existing widget
            item = self.network_presentation.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.network_presentation.insertWidget(0, self.network_widget)

        for i, (layer, neuron) in enumerate(self.current_neurons):
            self.network_widget.select_node(layer, neuron, self.neuron_colors[i])
            self.node_spin_boxes[i][0].setValue(layer)
            self.node_spin_boxes[i][1].setValue(neuron)

    def __on_change_algorithm(self, index: int):
        if Storage().algorithms:
            self.current_algorithm = Storage().algorithms[index].name

    def __handle_node_transition(self, selection_index: int, new_layer: int, new_node: int):
        """
        Handles the transition of a node from one layer/node to another.
        This includes updating the visual selection accordingly, as well as the stored data model.
        """
        old_layer, old_node = self.current_neurons[selection_index]

        if old_layer == new_layer and old_node == new_node:
            return

        if not self.network_widget:
            return

        # Check if the old node is used by another selection.
        # If so, the color of the selection underneath needs to be applied
        remaining_indices = [
            i for i, (l, n) in enumerate(self.current_neurons)
            if i != selection_index and l == old_layer and n == old_node
        ]
        if not remaining_indices:
            self.network_widget.unselect_node(old_layer, old_node)
        else:
            remaining_idx = remaining_indices[-1]
            self.network_widget.select_node(old_layer, old_node, self.neuron_colors[remaining_idx])

        self.current_neurons[selection_index] = (new_layer, new_node)

        self.network_widget.select_node(new_layer, new_node, self.neuron_colors[selection_index])

        self.neuron_selection_index = (selection_index + 1) % self.num_neurons

    def __on_change_layer_choice(self, selection_index: int, layer_index: int):
        if layer_index >= len(self.max_neuron_num_per_layer):
            return

        new_max_node_num = self.max_neuron_num_per_layer[layer_index]

        # Clip the node index to fit the new layer size
        current_node_val = self.current_neurons[selection_index][1]
        new_node_index = min(current_node_val, new_max_node_num - 1) if new_max_node_num > 0 else 0

        self.__handle_node_transition(selection_index, layer_index, new_node_index)

        # Update UI constraints for the Node SpinBox
        node_spin_box = self.node_spin_boxes[selection_index][1]
        node_spin_box.blockSignals(True)
        node_spin_box.setRange(0, new_max_node_num - 1 if new_max_node_num > 0 else 0)
        node_spin_box.setValue(new_node_index)
        node_spin_box.blockSignals(False)

    def __on_change_choice_within_layer(self, selection_index: int, node_index: int):
        current_layer = self.current_neurons[selection_index][0]
        self.__handle_node_transition(selection_index, current_layer, node_index)

    def __jump_to_neuron(self, neuron_index: int):
        if self.network_widget:
            layer, node = self.current_neurons[neuron_index]
            self.network_widget.go_to_node(layer, node)

    def __on_run_samples_clicked(self):
        if not Storage().networks:
            return
        if self.current_network < 0 or self.current_network >= len(Storage().networks):
            return
        config = Storage().networks[self.current_network]
        parent = self.parent()
        if parent is None or not hasattr(parent, "open_dialog"):
            return
        dialog = RunSamplesDialog(parent.close_dialog, config, on_results=lambda _res: self.__update_sample_results())
        parent.open_dialog(dialog)

    def __populate_bounds_selector(self, network_index: int):
        if self.bounds_selector is None:
            return
        self.bounds_selector.blockSignals(True)
        self.bounds_selector.clear()
        if network_index < 0 or network_index >= len(Storage().networks):
            self.bounds_selector.blockSignals(False)
            return
        config = Storage().networks[network_index]
        for i, _ in enumerate(config.saved_bounds):
            self.bounds_selector.addItem(f"Bounds {i + 1:02d}")
        if 0 <= config.selected_bounds_index < self.bounds_selector.count():
            self.bounds_selector.setCurrentIndex(config.selected_bounds_index)
        self.bounds_selector.blockSignals(False)

    def __on_bounds_changed(self, index: int):
        if self.current_network < 0 or self.current_network >= len(Storage().networks):
            return
        config = Storage().networks[self.current_network]
        config.selected_bounds_index = index
        self.__update_bounds_display()
        self.__update_sample_results()

    def __on_node_selection_change(self, layer_index: int, node_index: int) -> QColor | None:
        old_layer, old_node = self.current_neurons[self.neuron_selection_index]

        is_still_selected = any(
            (l == old_layer and n == old_node)
            for i, (l, n) in enumerate(self.current_neurons)
            if i != self.neuron_selection_index
        )

        if self.network_widget and not is_still_selected:
            self.network_widget.unselect_node(old_layer, old_node)

        self.current_neurons[self.neuron_selection_index] = (layer_index, node_index)

        layer_spin_box, value_spin_box = self.node_spin_boxes[self.neuron_selection_index]

        layer_spin_box.blockSignals(True)
        value_spin_box.blockSignals(True)

        layer_spin_box.setValue(layer_index)
        value_spin_box.setValue(node_index)

        layer_spin_box.blockSignals(False)
        value_spin_box.blockSignals(False)

        former_index = self.neuron_selection_index
        self.neuron_selection_index = (self.neuron_selection_index + 1) % self.num_neurons

        return self.neuron_colors[former_index]

    def __get_side_bar_content(self) -> QVBoxLayout:
        layout = QVBoxLayout()

        title = QLabel("Neuron Picker")
        title.setObjectName("title")

        layout.addWidget(title)

        # --- Network Selector ---
        network_group = QHBoxLayout()
        network_group.addWidget(QLabel("Network:"))
        network_group.addWidget(self.network_selector)

        # --- Algorithm Selector ---
        algorithm_group = QHBoxLayout()
        algorithm_group.addWidget(QLabel("Algorithm:"))
        self.update_algorithms()
        algorithm_group.addWidget(self.algorithm_selector)

        layout.addLayout(network_group)
        layout.addLayout(algorithm_group)

        # --- Bounds Selector ---
        bounds_group = QHBoxLayout()
        bounds_group.setSpacing(8)
        bounds_group.addWidget(QLabel("Bounds:"))
        self.bounds_selector = QComboBox()
        self.__populate_bounds_selector(self.current_network)
        self.bounds_selector.currentIndexChanged.connect(self.__on_bounds_changed)
        bounds_group.addWidget(self.bounds_selector)
        self.bounds_toggle_button = QPushButton("👁")
        self.bounds_toggle_button.setObjectName("transparent-button")
        self.bounds_toggle_button.setFixedWidth(32)
        self.bounds_toggle_button.clicked.connect(self.__toggle_bounds_display)
        self.bounds_toggle_button.setVisible(True)
        bounds_group.addWidget(self.bounds_toggle_button)
        layout.addLayout(bounds_group)
        layout.addSpacing(8)

        # --- Bounds Display ---
        self.bounds_display_group = BoundsDisplayWidget("Bounds", scrollable=True, min_height=200, max_height=260)
        input_count = 0
        if Storage().networks:
            input_count = Storage().networks[self.current_network].layers_dimensions[0]
        self._bounds_index_label_width = self.__compute_bounds_index_label_width(input_count)
        self.bounds_display_group.set_rows(input_count, index_label_width=self._bounds_index_label_width)
        self.__rebuild_bounds_display_rows()

        # --- Sample Results ---
        # --- Neuron Pair Selectors ---
        for i in range(0, self.num_neurons):
            neuron_group = QHBoxLayout()
            label = QLabel(f"Neuron {i + 1}")
            label.setAlignment(Qt.AlignmentFlag.AlignBottom)
            neuron_group.addWidget(label)

            # Color Circle
            color = self.neuron_colors[i].name()
            color_circle = QLabel()
            color_circle.setFixedSize(16, 16)
            color_circle.setStyleSheet(f"background-color: {color}; border-radius: 8px;")
            neuron_group.addWidget(color_circle, alignment=Qt.AlignmentFlag.AlignBottom)

            # Controls
            layer_spinbox = QSpinBox()
            layer_spinbox.setFixedWidth(48)
            layer_spinbox.setAlignment(Qt.AlignmentFlag.AlignLeft)

            neuron_spinbox = QSpinBox()
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

            eye_button = QPushButton("")
            eye_button.setIcon(QIcon(":/assets/icons/focus_node.svg"))
            eye_button.setObjectName("icon-button")
            eye_button.clicked.connect(lambda _, idx=i: self.__jump_to_neuron(idx))

            layer_box = QVBoxLayout()
            layer_box.addWidget(layer_hint)
            layer_box.addWidget(layer_spinbox)
            layer_box.setAlignment(Qt.AlignmentFlag.AlignBottom)

            node_box = QVBoxLayout()
            node_box.addWidget(node_hint)
            node_box.addWidget(neuron_spinbox)
            node_box.setAlignment(Qt.AlignmentFlag.AlignBottom)

            neuron_group.addStretch()
            neuron_group.addLayout(layer_box)
            neuron_group.addWidget(colon)
            neuron_group.addLayout(node_box)
            neuron_group.addWidget(eye_button, alignment=Qt.AlignmentFlag.AlignBottom)

            layout.addLayout(neuron_group)

        layout.addSpacing(12)
        layout.addWidget(self.bounds_display_group)
        self.bounds_display_group.setVisible(False)
        self.__update_bounds_display()

        self.sample_metrics = SampleMetricsWidget(
            "Sample Results",
            include_min=False,
            max_items=10,
            scrollable=False,
        )
        layout.addSpacing(12)
        self.sample_metrics.setVisible(False)
        layout.addWidget(self.sample_metrics)
        self.full_results_button = QPushButton("Full Results")
        self.full_results_button.setVisible(False)
        self.full_results_button.setEnabled(False)
        self.full_results_button.clicked.connect(self.__on_full_results_clicked)
        layout.addWidget(self.full_results_button)

        layout.addStretch()

        print("Sidebar loaded")
        return layout

    def __update_bounds_display(self):
        if self.bounds_display_group is None:
            return
        if self.current_network < 0 or self.current_network >= len(Storage().networks):
            self.bounds_display_group.setTitle("Bounds")
            self.bounds_display_group.set_values(None)
            return
        config = Storage().networks[self.current_network]
        index = config.selected_bounds_index
        if index < 0 or index >= len(config.saved_bounds):
            self.bounds_display_group.setTitle("Bounds")
            self.bounds_display_group.set_values(None)
            self.__update_sample_results()
            return
        bounds = config.saved_bounds[index]
        self.bounds_display_group.setTitle(f"Bounds {index + 1:02d}")
        self.bounds_display_group.set_values(bounds.get_values())
        self.__update_sample_results()

    def __toggle_bounds_display(self):
        if self.bounds_display_group is None:
            return
        self.bounds_display_group.setVisible(not self.bounds_display_group.isVisible())

    def __update_sample_results(self):
        if self.sample_metrics is None:
            return
        if not Storage().networks:
            self.sample_metrics.set_result(None)
            self.sample_metrics.setVisible(False)
            self.full_results_button.setEnabled(False)
            self.full_results_button.setVisible(False)
            return
        config = Storage().networks[self.current_network]
        index = getattr(config, "selected_bounds_index", -1)
        if index < 0 or index >= len(config.saved_bounds):
            self.sample_metrics.set_result(None)
            self.sample_metrics.setVisible(False)
            self.full_results_button.setEnabled(False)
            self.full_results_button.setVisible(False)
            return
        bounds = config.saved_bounds[index]
        result = bounds.get_sample()
        self.sample_metrics.set_result(result)
        self.sample_metrics.setVisible(result is not None)
        self.full_results_button.setEnabled(result is not None)
        self.full_results_button.setVisible(result is not None)

    def __on_full_results_clicked(self):
        if not Storage().networks:
            return
        config = Storage().networks[self.current_network]
        index = getattr(config, "selected_bounds_index", -1)
        if index < 0 or index >= len(config.saved_bounds):
            return
        result = config.saved_bounds[index].get_sample()
        if result is None:
            return
        parent = self.parent()
        if parent is None or not hasattr(parent, "open_dialog"):
            return
        parent.open_dialog(SampleResultsDialog(parent.close_dialog, result))

    def __rebuild_bounds_display_rows(self):
        if self.bounds_display_group is None:
            return
        input_count = 0
        if Storage().networks and 0 <= self.current_network < len(Storage().networks):
            input_count = Storage().networks[self.current_network].layers_dimensions[0]
        self._bounds_index_label_width = self.__compute_bounds_index_label_width(input_count)
        self.bounds_display_group.set_rows(input_count, index_label_width=self._bounds_index_label_width)
