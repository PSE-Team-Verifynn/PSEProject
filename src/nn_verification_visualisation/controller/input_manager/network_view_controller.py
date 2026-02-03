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

if TYPE_CHECKING:
    from nn_verification_visualisation.view.network_view.network_view import NetworkView


class NetworkViewController:
    current_network_view: NetworkView
    current_tab: int

    def __init__(self, current_network_view: NetworkView):
        self.current_network_view = current_network_view
        self.current_tab = 0

    def open_network_view(self, network: NeuralNetwork) -> bool:
        pass

    def open_network_management_dialog(self):
        dialog = NetworkManagementDialog(self)
        self.current_network_view.open_dialog(dialog)

    def load_bounds(self, config: NetworkVerificationConfig) -> bool:
        path = self.current_network_view.open_network_file_picker("Bound-Files (*.csv *.vnnlib);; All Files (*)")
        if path is None:
            return False
        result = InputBoundsLoader().load_input_bounds(path, config)
        if result.is_success:
            config.bounds.load_bounds(result.data)

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
        return True

    def run_samples(self) -> List[int]:
        pass

    def add_sample(self, bounds: InputBounds) -> List[int]:
        pass

    def change_tab(self, index: int):
        pass