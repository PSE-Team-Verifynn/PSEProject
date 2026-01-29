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

    def load_save_state(self, save_state: SaveState):
        pass

    def get_save_state(self) -> SaveState:
        pass

    def remove_algorithm(self, algo_name):
        self.algorithms.remove(algo_name)
        self.__call_listeners()

    def modify_algorithm(self, algo_name, new_algorithm):
        self.algorithms[algo_name] = new_algorithm
        self.__call_listeners()

    def add_algorithm(self, new_algorithm):
        self.algorithms.append(new_algorithm)
        self.__call_listeners()

    def __call_listeners(self):
        for listener in self.algorithm_change_listeners:
            listener()