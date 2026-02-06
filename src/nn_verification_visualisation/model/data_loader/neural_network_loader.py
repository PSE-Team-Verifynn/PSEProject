from pathlib import Path

from nn_verification_visualisation.utils.result import *
from nn_verification_visualisation.utils.singleton import SingletonMeta
from nn_verification_visualisation.model.data.neural_network import NeuralNetwork

import onnx

class NeuralNetworkLoader(metaclass=SingletonMeta):
    """
    Class to load neural network model.
    """
    def load_neural_network(self, file_path: str) -> Result[NeuralNetwork]:
        """
        Function to load neural network model.
        :param file_path: path to neural network model.
        :return: instance of neural network model and result as success or failure.
        """
        try:
            model = onnx.load_model(file_path)
            onnx.checker.check_model(model, full_check=True)
            return Success(NeuralNetwork(Path(file_path).stem, file_path, model))
        except BaseException as e:
            return Failure(e)