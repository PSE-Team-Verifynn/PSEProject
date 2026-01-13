from typing import List

from onnx import ModelProto

class NetworkModifier:
    def custom_output_layer(self, model: ModelProto, neurons: List[(int, int)]) -> ModelProto:
        pass