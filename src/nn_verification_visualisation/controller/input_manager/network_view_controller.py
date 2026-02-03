from __future__ import annotations

from typing import List, TYPE_CHECKING

import onnx

from nn_verification_visualisation.controller.process_manager.network_modifier import NetworkModifier
from nn_verification_visualisation.model.data.input_bounds import InputBounds
from nn_verification_visualisation.model.data.network_verification_config import NetworkVerificationConfig
from nn_verification_visualisation.model.data.neural_network import NeuralNetwork
from nn_verification_visualisation.model.data.storage import Storage
from nn_verification_visualisation.model.data_loader.input_bounds_loader import InputBoundsLoader
from nn_verification_visualisation.model.data_loader.neural_network_loader import NeuralNetworkLoader
from nn_verification_visualisation.view.dialogs.info_popup import InfoPopup
from nn_verification_visualisation.view.dialogs.info_type import InfoType
from nn_verification_visualisation.view.dialogs.network_management_dialog import NetworkManagementDialog
from nn_verification_visualisation.view.dialogs.run_samples_dialog import RunSamplesDialog

if TYPE_CHECKING:
    from nn_verification_visualisation.view.network_view.network_view import NetworkView


class NetworkViewController:
    current_network_view: NetworkView
    current_tab: int

    def __init__(self, current_network_view: NetworkView):
        self.current_network_view = current_network_view
        self.current_tab = 0
        self._draft_bounds_by_config: dict[NetworkVerificationConfig, list[tuple[float, float]]] = {}

    def open_network_view(self, network: NeuralNetwork) -> bool:
        pass

    def open_network_management_dialog(self):
        dialog = NetworkManagementDialog(self)
        self.current_network_view.open_dialog(dialog)

    def open_run_samples_dialog(self, config: NetworkVerificationConfig):
        dialog = RunSamplesDialog(self.current_network_view.close_dialog, config)
        self.current_network_view.open_dialog(dialog)

    def load_bounds(self, config: NetworkVerificationConfig) -> bool:
        path = self.current_network_view.open_network_file_picker("Bound-Files (*.csv *.vnnlib);; All Files (*)")
        if path is None:
            return False
        result = InputBoundsLoader().load_input_bounds(path, config)
        if result.is_success:
            self._apply_loaded_bounds(config, result.data)

        if result.is_success:
            dialog_type = InfoType.CONFIRMATION
            text = "Bounds were loaded successfully!"
        else:
            dialog_type = InfoType.ERROR
            text = repr(result.error)

        self.current_network_view.open_dialog(InfoPopup(self.current_network_view.close_dialog, text, dialog_type))
        return result.is_success

    def load_new_network(self) -> NetworkVerificationConfig | None:
        path = self.current_network_view.open_network_file_picker("ONNX-Files (*.onnx);; All Files (*)")
        if path is None:
            return None
        result = NeuralNetworkLoader().load_neural_network(path)
        if not result.is_success:
            return None
        layer_dimensions = []   # list of the number of nodes per Layer
        for layer in result.data.model.graph.initializer: # adds the 1.dim of the matrix, dim of the 1. layer
            if len(layer.dims) == 2 :
                layer_dimensions.append(layer.dims[0])
        layer_dimensions.append(result.data.model.graph.output[0].type.tensor_type.shape.dim[1].dim_value)# adds the output layer dim

        network = NetworkVerificationConfig(result.data,layer_dimensions)   #layer_dimensions is used for visualization of the network

        storage = Storage()
        storage.networks.append(network)

        self.current_network_view.add_network_tab(network)
        self.current_tab = len(storage.networks)

        return network

    def remove_neural_network(self, network: NetworkVerificationConfig) -> bool:
        networks = Storage().networks
        if network not in networks:
            return False
        index = networks.index(network)
        self.current_network_view.close_network_tab(index)
        networks.remove(network)
        self._draft_bounds_by_config.pop(network, None)
        return True

    def run_samples(self) -> List[int]:
        pass

    def add_sample(self, bounds: InputBounds) -> List[int]:
        pass

    def change_tab(self, index: int):
        pass

    def save_bounds(self, config: NetworkVerificationConfig) -> int:
        bounds = config.bounds.get_values()
        saved = InputBounds(config.layers_dimensions[0])
        saved.load_list(bounds)
        saved.set_read_only(True)
        saved.clear_sample()
        config.saved_bounds.append(saved)
        config.selected_bounds_index = len(config.saved_bounds) - 1
        self._set_draft_bounds(config, [(0.0, 0.0)] * config.layers_dimensions[0])
        config.bounds.set_read_only(True)
        return config.selected_bounds_index

    def select_bounds(self, config: NetworkVerificationConfig, bounds_index: int | None):
        if bounds_index is None:
            config.selected_bounds_index = -1
            config.bounds.load_list(self._get_draft_bounds(config))
            config.bounds.set_read_only(False)
            config.bounds.clear_sample()
            return

        if bounds_index < 0 or bounds_index >= len(config.saved_bounds):
            return

        if config.selected_bounds_index == -1:
            self._set_draft_bounds(config, config.bounds.get_values())

        config.selected_bounds_index = bounds_index
        saved = config.saved_bounds[bounds_index]
        config.bounds.load_list(saved.get_values())
        config.bounds.set_read_only(True)
        config.bounds.clear_sample()

    def remove_bounds(self, config: NetworkVerificationConfig, bounds_index: int) -> bool:
        if bounds_index < 0 or bounds_index >= len(config.saved_bounds):
            return False
        del config.saved_bounds[bounds_index]
        if config.selected_bounds_index == bounds_index:
            self.select_bounds(config, None)
            return True
        if config.selected_bounds_index > bounds_index:
            config.selected_bounds_index -= 1
        return True

    def _apply_loaded_bounds(self, config: NetworkVerificationConfig, bounds: dict[int, tuple[float, float]]):
        bounds_list = [bounds[i] for i in range(config.layers_dimensions[0])]
        if config.selected_bounds_index == -1:
            config.bounds.load_list(bounds_list)
            self._set_draft_bounds(config, bounds_list)
            config.bounds.set_read_only(False)
            config.bounds.clear_sample()
            return

        saved = config.saved_bounds[config.selected_bounds_index]
        saved.load_list(bounds_list)
        saved.clear_sample()
        config.bounds.load_list(bounds_list)
        config.bounds.set_read_only(True)

    def _get_draft_bounds(self, config: NetworkVerificationConfig) -> list[tuple[float, float]]:
        if config not in self._draft_bounds_by_config:
            self._draft_bounds_by_config[config] = config.bounds.get_values()
        return self._draft_bounds_by_config[config]

    def _set_draft_bounds(self, config: NetworkVerificationConfig, bounds: list[tuple[float, float]]):
        self._draft_bounds_by_config[config] = bounds
