from typing import List

from nn_verification_visualisation.model.data.algorithm import Algorithm
from nn_verification_visualisation.model.data.network_verification_config import NetworkVerificationConfig


class PlotGenerationConfig:
    '''
    Stores a single selection of a network, an algorithm, a neuron tuple and parameters made in the NeuronPicker.
    :param nnconfig: the selected network with its input bounds.
    :param algorithm: the selected algorithm.
    :param selected_neurons: the selected neuron tuple.
    :param parameters: the selected parameters for the algorithm execution.
    '''
    nnconfig: NetworkVerificationConfig
    algorithm: Algorithm
    selected_neurons: List[tuple[int, int]]
    parameters: List[str]

    def __init__(self, nnconfig: NetworkVerificationConfig, algorithm: Algorithm, selected_neurons: List[tuple[int, int]], parameters: List[str]):
        self.nnconfig = nnconfig
        self.algorithm = algorithm
        self.selected_neurons = selected_neurons
        self.parameters = parameters