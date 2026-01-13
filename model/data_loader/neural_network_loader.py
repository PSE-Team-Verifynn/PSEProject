from utils.result import Result
from utils.singleton import SingletonMeta
from model.data.neural_network import NeuralNetwork

class NeuralNetworkLoader(metaclass=SingletonMeta):
    def load_neural_network(self, file_path: str) -> Result[NeuralNetwork]:
        pass