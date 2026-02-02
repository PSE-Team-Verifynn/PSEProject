from typing import List

from nn_verification_visualisation.model.data.input_bounds import InputBounds
from nn_verification_visualisation.model.data.neural_network import NeuralNetwork


class NetworkVerificationConfig:
    network: NeuralNetwork
    bounds: InputBounds
    saved_bounds: List[InputBounds]
    selected_bounds_index: int
    activation_values: List[int]
    layers_dimensions: List[int]

    def __init__(self, network: NeuralNetwork, layers_dimensions: List[int]):
        self.network = network
        self.layers_dimensions = layers_dimensions
        self.bounds = InputBounds(layers_dimensions[0])
        self.saved_bounds = []
        self.selected_bounds_index = -1
