from typing import List

from model.data.input_bounds import InputBounds
from model.data.neural_network import NeuralNetwork


class NetworkVerificationConfig:
    network: NeuralNetwork
    bounds: InputBounds
    activation_values: List[int]