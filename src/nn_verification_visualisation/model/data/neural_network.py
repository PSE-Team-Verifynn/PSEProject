from onnx import ModelProto

class NeuralNetwork:
    '''
    Data object for a neural network. Stores the name and path of the network file, as well as the model itself.
    :param name: the name of the network file.
    :param path: the path of the network file.
    :param model: the complete model of the network provided by onnx.
    '''
    name: str
    path: str
    model: ModelProto

    def __init__(self, name: str, path: str, model: ModelProto):
        self.name = name
        self.model = model
        self.path = path
