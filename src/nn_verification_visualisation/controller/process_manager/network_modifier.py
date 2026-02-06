import copy

from onnx import ModelProto, TensorProto, NodeProto
import onnx


class NetworkModifier:
    @staticmethod
    def with_all_outputs(
        static_model: ModelProto,
        sampling_mode: str = "pre_activation_after_bias",
    ) -> ModelProto:
        model = copy.deepcopy(static_model)
        existing = {output.name for output in model.graph.output}
        initializer_names = {init.name for init in model.graph.initializer if init.name}
        producer_by_output: dict[str, NodeProto] = {}
        for node in model.graph.node:
            for output_name in node.output:
                if output_name:
                    producer_by_output[output_name] = node
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
            if sampling_mode == "pre_activation_before_bias":
                if node.input and node.input[0]:
                    pre_activation = NetworkModifier._resolve_pre_activation_before_bias(
                        node.input[0],
                        producer_by_output,
                        initializer_names,
                    )
                    names.append(pre_activation)
            elif sampling_mode == "pre_activation_after_bias":
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

    @staticmethod
    def _resolve_pre_activation_before_bias(
        activation_input: str,
        producer_by_output: dict[str, NodeProto],
        initializer_names: set[str],
    ) -> str:
        producer = producer_by_output.get(activation_input)
        if producer is None or producer.op_type != "Add" or len(producer.input) < 2:
            return activation_input
        left, right = producer.input[0], producer.input[1]
        left_is_bias = left in initializer_names
        right_is_bias = right in initializer_names
        if left_is_bias and not right_is_bias:
            return right
        if right_is_bias and not left_is_bias:
            return left
        return activation_input

    def custom_output_layer(self, static_model: ModelProto, neurons: list[tuple[int, int]], directions: list[tuple[float, float]]) -> ModelProto:
        '''

        :param static_model: the whole network, which is not changed in this function
        :param neurons: List of neurons, that should be used for the calculation
        :param directions: List of directions, that represent linear combinations of neurons
        :return: the new model
        '''
        model = copy.deepcopy(static_model)     #deepcopies so the original model does not change
        model.graph.node[model.graph.node.__len__() - 1].output.remove("output")
        model.graph.node[model.graph.node.__len__() - 1].output.append("old_output")    #redirects the old output layer
        initializers = NetworkModifier.create_initalizers(self, model, neurons, directions)
        model.graph.initializer.append(initializers[0])
        model.graph.initializer.append(initializers[1])     # adds the new initializers
        new_node = NetworkModifier.create_new_layer(self, model, neurons, initializers)
        model.graph.node.append(new_node)   # adds the new node
        model = NetworkModifier.add_bridge_neurons(self, model, neurons, directions)
        model.graph.output[0].type.tensor_type.shape.dim[1].dim_value =  directions.__len__()       #modifies the output dim, so it matches with the initializers
        return model

    def add_bridge_neurons(self, model: ModelProto, neurons: list[tuple[int, int]], directions: list[tuple[float, float]]) -> ModelProto:
        '''

        :param model: the whole network
        :param neurons: List of neurons, that should be used for the calculation
        :param directions: List of directions, that represent linear combinations of neurons
        :return: the modified model
        '''
        for neuron in neurons:
            for layer in range(2 * neuron[0], model.graph.initializer.__len__() - 1):   # goes through all layers following
                                                                                        # the layer with the neuron in it and adds a new neuron for each layer
                if layer != 2 * neuron[0]:
                    model.graph.initializer[layer].dims[0] += 1
                if model.graph.initializer[layer].dims.__len__() == 2 and layer != model.graph.initializer.__len__() - 2:
                    model.graph.initializer[layer].dims[1] += 1
                    if layer == 2 * neuron[0]:
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
                            model.graph.initializer[layer].float_data.append(0)
                        model.graph.initializer[layer].float_data.append(1)
                else:
                    if layer != model.graph.initializer.__len__() - 2:
                        model.graph.initializer[layer].float_data.append(0)
        for neuron_ind in range(0, neurons.__len__()): # changes the last initializer to match the output
            for direction in directions:
                model.graph.initializer[model.graph.initializer.__len__() - 2].float_data.append(direction[neuron_ind])
        return model

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
        new_node.output.append("output")
        new_node.name = "new_output"
        return  new_node
