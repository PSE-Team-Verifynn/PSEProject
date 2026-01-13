from typing import List

from model.data.input_bounds import InputBounds
from model.data.neural_network import NeuralNetwork
from view.network_view.network_view import NetworkView


class NetworkViewController:
    current_network_view: NetworkView
    current_tab: int

    def __init__(self, current_network_view: NetworkView):
        self.current_network_view = current_network_view
        self.current_tab = 0

    def open_network_view(self, network: NeuralNetwork) -> bool:
        pass

    def load_new_network(self):
        pass

    def run_samples(self) -> List[int]:
        pass

    def add_sample(self, bounds: InputBounds) -> List[int]:
        pass

    def change_tab(self, index: int):
        pass