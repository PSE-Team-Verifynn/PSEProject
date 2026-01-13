from typing import List

from model.data.algorithm import Algorithm
from model.data.network_verification_config import NetworkVerificationConfig


class PlotGenerationConfig:
    nnconfig: NetworkVerificationConfig
    algorithm: Algorithm
    selected_neurons: List[(int, int)]
    parameters: List[str]

    def __init__(self, nnconfig: NetworkVerificationConfig, algorithm: Algorithm, selected_neurons: List[(int, int)], parameters: List[str]):
        self.nnconfig = nnconfig
        self.algorithm = algorithm
        self.selected_neurons = selected_neurons
        self.parameters = parameters