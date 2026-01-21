from __future__ import annotations

from typing import List, TYPE_CHECKING

from nn_verification_visualisation.model.data.input_bounds import InputBounds
from nn_verification_visualisation.model.data.network_verification_config import NetworkVerificationConfig
from nn_verification_visualisation.model.data.neural_network import NeuralNetwork
from nn_verification_visualisation.model.data.storage import Storage
from nn_verification_visualisation.model.data_loader.neural_network_loader import NeuralNetworkLoader
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

    def load_new_network(self) -> NetworkVerificationConfig | None:
        path = self.current_network_view.open_network_file_picker()
        if path is None:
            return None

        result = NeuralNetworkLoader().load_neural_network(path)
        layer_dimensions = []   #list of the number of nodes per Layer
        for layer in result.data.model.graph.initializer:
            layer_dimensions.append(layer.dims[0])
        if not result.is_success:
            return None

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