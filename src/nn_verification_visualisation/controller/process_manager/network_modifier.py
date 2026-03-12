import copy

import numpy as np
from onnx import ModelProto, TensorProto, NodeProto
import onnx
from sympy import true
from sympy.codegen.ast import none


class NetworkModifier:
    """
    Class to modify a network.
    """

    @staticmethod
    def with_all_outputs(
        static_model: ModelProto,
        sampling_mode: str = "pre_activation_after_bias",
    ) -> ModelProto:
        model = copy.deepcopy(static_model)
        existing = {output.name for output in model.graph.output}
        activation_ops = {
            "Relu",
            "Sigmoid",
            "Tanh",
            "Softmax",
            "LogSoftmax",
            "LeakyRelu",
            "Elu",
            "Gelu",
            "Clip",
            "HardSigmoid",
            "HardSwish",
            "PRelu",
            "Selu",
            "Celu",
            "Mish",
            "Softplus",
            "Softsign",
            "Swish",
        }
        for node in model.graph.node:
            if node.op_type not in activation_ops:
                continue
            names: list[str] = []
            if sampling_mode == "pre_activation_after_bias":
                if node.input and node.input[0]:
                    names.append(node.input[0])  # pre-activation after bias
            elif sampling_mode == "post_activation":
                if node.output and node.output[0]:
                    names.append(node.output[0])  # post-activation
            else:
                raise ValueError(f"Invalid sampling_mode: {sampling_mode}")
            for name in names:
                if not name or name in existing:
                    continue
                vi = onnx.ValueInfoProto()
                vi.name = name
                model.graph.output.append(vi)
                existing.add(name)
        return model

    def custom_output_layer(self, static_model: ModelProto, neurons: list[tuple[int, int]], directions: list[tuple[float, float]]) -> ModelProto:
        '''

        :param static_model: the whole network, which is not changed in this function
        :param neurons: List of neurons, that should be used for the calculation
        :param directions: List of directions, that represent linear combinations of neurons
        :return: the new model
        '''
        model = copy.deepcopy(static_model)     #deepcopies so the original model does not change

        output_names = model.graph.node[-1].output
        if output_names.__len__() != 1:
            raise RuntimeError("The last layer of the network must have exactly one output")

        model.graph.node[model.graph.node.__len__() - 1].output.remove(output_names[0])
        model.graph.node[model.graph.node.__len__() - 1].output.append("old_output")            #redirects the old output layer
        model = NetworkModifier.change_initialiser_data_format(self, model)
        initializers = NetworkModifier.create_initalizers(self, model, neurons, directions)
        model.graph.initializer.append(initializers[0])
        model.graph.initializer.append(initializers[1])                                         # adds the new initializers
        new_node = NetworkModifier.create_new_layer(self, model, neurons, initializers)
        model.graph.node.append(new_node)                                                       # adds the new node
        model = NetworkModifier.add_bridge_neurons(self, model, neurons, directions)
        model.graph.output[0].type.tensor_type.shape.dim[-1].dim_value =  directions.__len__()  #modifies the output dim, so it matches with the initializers
        onnx.save_model(model, "Test6","textproto", save_as_external_data=true)

        return model

    def change_initialiser_data_format(self, model:ModelProto) -> ModelProto:
        '''
        This method forces the data format to be float data
        :param model: the model that is changed
        :return: the model with the correct data format
        '''
        for element in model.graph.initializer :
            numpy_initializer = onnx.numpy_helper.to_array(element, np.float32)                 # saves the data as a numpy array
            name = element.name
            element.Clear()
            for entry in np.nditer(numpy_initializer):
                element.float_data.append(float(entry))                                        # adds the data back, forced as float
            for dim in numpy_initializer.shape:
                element.dims.append(dim)
            element.data_type = 1
            element.name = name
        return model

    @staticmethod
    def add_bridge_neurons(self, model: ModelProto, neurons: list[tuple[int, int]], directions: list[tuple[float, float]]) -> ModelProto:
        '''

        :param model: the whole network
        :param neurons: List of neurons, that should be used for the calculation
        :param directions: List of directions, that represent linear combinations of neurons
        :return: the modified model
        '''
        offset = 0
        if model.graph.initializer[0].dims.__len__() > 3:                     # adds an offset to the layer iteration if a preprocess layer exists (problem if there are more)
            offset = 1
        dirty_trick_constant = 10000                                           # this is the constant for fooling relu on bridge neurons
        for neuron in neurons:
            for layer in range(2 * neuron[0] + offset, model.graph.initializer.__len__() - 1):   # goes through all layers following
                if model.graph.initializer[layer].dims.__len__() < 3:
                    if layer != 2 * neuron[0] + offset:
                        model.graph.initializer[layer].dims[0] += 1                      #changes the dims off all following layers to the layer of the selected neuron
                    if model.graph.initializer[layer].dims.__len__() == 2 and layer != model.graph.initializer.__len__() - 2:           # adds connections to Matrix layers
                        model.graph.initializer[layer].dims[1] += 1                     # changes the 2 dim for the matrix multiplication
                        if layer == 2 * neuron[0] + offset:
                            for node in range(0, model.graph.initializer[layer].dims[0]):
                                if node == neuron[1]:
                                    model.graph.initializer[layer].float_data.insert(                       # adds the connections between the new neurons
                                        (node + 1) * model.graph.initializer[layer].dims[1] - 1, 1)
                                else:
                                    model.graph.initializer[layer].float_data.insert(                       # adds the connections between old and new neurons
                                        (node + 1) * model.graph.initializer[layer].dims[1] - 1, 0)
                        else:
                            for node in range(1, model.graph.initializer[layer].dims[0] - 1):
                                model.graph.initializer[layer].float_data.insert(               # adds the connections to the new neurons
                                    node * model.graph.initializer[layer].dims[1] - 1, 0)
                            model.graph.initializer[layer].float_data.append(0)
                            for node in range(1, model.graph.initializer[layer].dims[1]):
                                model.graph.initializer[layer].float_data.append(0)             # adds connections from new neurons to neurons in the next layer
                            model.graph.initializer[layer].float_data.append(1)                 # gives over the value to the new bridge neuron
                    else:                                                                       # adds the new biases for the new neurons
                        if layer != model.graph.initializer.__len__() - 2:
                            if layer == 2 * neuron[0] + 1 +offset and layer == model.graph.initializer.__len__() - 3:
                                model.graph.initializer[layer].float_data.append(0)
                            elif layer == 2 * neuron[0] + 1 + offset:
                                model.graph.initializer[layer].float_data.append(dirty_trick_constant)          # dirty trick start
                            elif layer == model.graph.initializer.__len__() - 3 :
                                model.graph.initializer[layer].float_data.append(-1 * dirty_trick_constant)     # dirty trick end
                            else:
                                model.graph.initializer[layer].float_data.append(0)
        for neuron_ind in range(0, neurons.__len__()):          # changes the last initializer to match the output
            if 2 * neurons[neuron_ind][0] == model.graph.initializer.__len__() - 2:
                for direction in range(0, directions.__len__()):                    # adds the values of the directions into the last matrix multiplication
                    model.graph.initializer[model.graph.initializer.__len__() - 2].float_data.remove(0)
                    model.graph.initializer[model.graph.initializer.__len__() - 2].float_data.insert(2* neurons[neuron_ind][1] + direction,directions[direction][neuron_ind])
            else:
                for direction in directions:
                    model.graph.initializer[model.graph.initializer.__len__() - 2].float_data.append(direction[neuron_ind])
        return model

    @staticmethod
    def create_initalizers(self, model: ModelProto, neurons: list[tuple[int, int]], directions: list[tuple[float, float]]) -> tuple[TensorProto, TensorProto]:
        '''

        :param model: the whole network
        :param neurons: List of neurons, that should be used for the calculation
        :param directions: List of directions, that represent linear combinations of neurons
        :return: returns the new initializers for the new output layer
        '''
        new_initializer1 = copy.deepcopy(model.graph.initializer[0])
        new_initializer2 = copy.deepcopy(model.graph.initializer[0])
        new_initializer1.name = "output_initializer_W"
        new_initializer2.name = "output_initializer_B"
        del new_initializer1.dims[:]
        del new_initializer2.dims[:]
        del new_initializer1.float_data[:]
        del new_initializer2.float_data[:]      #creates the 2 new initalizers, without data
        new_initializer1.dims.append(model.graph.initializer[model.graph.initializer.__len__() - 1].dims[0])
        new_initializer1.dims.append(directions.__len__())
        new_initializer2.dims.append(directions.__len__())
        for i in range(new_initializer1.dims[0] * new_initializer1.dims[1]):    #adds data, so that dims matches the number of elements
            new_initializer1.float_data.append(0)
        for i in range(new_initializer2.dims[0]):
            new_initializer2.float_data.append(0)
        return new_initializer1, new_initializer2

    @staticmethod
    def create_new_layer(self, model: ModelProto, neurons: list[tuple[int, int]], initializers: tuple[TensorProto, TensorProto]) -> NodeProto:
        '''
        :param model: the whole network
        :param neurons: List of neurons, that should be used for the calculation
        :param initializers: the initializers for the new output layer
        :return: the new output layer
        '''
        new_node = copy.deepcopy(model.graph.node[0])
        del new_node.input[:]
        del new_node.output[:]      #creates the new emtpy layer
        new_node.input.append(model.graph.node[model.graph.node.__len__() - 1].output[0])   #adds the data
        new_node.input.append(initializers[0].name)
        new_node.input.append(initializers[1].name)
        new_node.output.append(model.graph.output[0].name)
        new_node.name = "new_output"
        new_node.op_type = "Gemm"
        return  new_node
