from typing import Dict
from onnx import ModelProto

class NeuralNetwork:
    name: str
    path: str
    model: ModelProto

    def __init__(self, name: str, path: str, model: ModelProto):
        self.name = name
        self.model = model
        self.path = path
