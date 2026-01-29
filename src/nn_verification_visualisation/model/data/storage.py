from typing import List, Callable

from nn_verification_visualisation.model.data.algorithm import Algorithm
from nn_verification_visualisation.model.data.diagram_config import DiagramConfig
from nn_verification_visualisation.model.data.network_verification_config import NetworkVerificationConfig
from nn_verification_visualisation.model.data.save_state import SaveState
from nn_verification_visualisation.utils.singleton import SingletonMeta


class Storage(metaclass=SingletonMeta):
    networks: List[NetworkVerificationConfig]
    diagrams: List[DiagramConfig]
    algorithms: List[Algorithm]
    algorithm_change_listeners: List[Callable[[], None]]

    def __init__(self):
        self.networks = []
        self.diagrams = []
        self.algorithms = []
        self.algorithm_change_listeners = []

    def load_save_state(self, save_state: SaveState):
        pass

    def get_save_state(self) -> SaveState:
        pass

    def remove_algorithm(self, algo_path):
        matching_indeces = [i for i in range(len(self.algorithms)) if self.algorithms[i].path == algo_path]
        if not matching_indeces:
            return
        del self.algorithms[matching_indeces[0]]
        self.__call_listeners()

    def modify_algorithm(self, algo_path, new_algorithm):
        matching_indices = [i for i in range(len(self.algorithms)) if self.algorithms[i].path == algo_path]
        if not matching_indices:
            return
        self.algorithms[matching_indices[0]] = new_algorithm
        self.__call_listeners()

    def add_algorithm(self, new_algorithm):
        self.algorithms.append(new_algorithm)
        self.__call_listeners()

    def __call_listeners(self):
        for listener in self.algorithm_change_listeners:
            listener()