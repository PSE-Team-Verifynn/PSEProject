from utils.result import Result
from utils.singleton import SingletonMeta

from model.data.neural_network import NeuralNetwork
from model.data.input_bounds import InputBounds

class InputBoundsLoader(metaclass=SingletonMeta):
    def load_input_bounds(self, file_path: str, network: NeuralNetwork) -> Result[InputBounds]:
        pass