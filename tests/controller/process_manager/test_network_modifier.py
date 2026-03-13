import copy

import numpy as np
from onnx import ModelProto, TensorProto, NodeProto
import onnx
from sympy import true
import TestFiles
from nn_verification_visualisation.controller.process_manager.algorithm_executor import AlgorithmExecutor
from nn_verification_visualisation.controller.process_manager.network_modifier import NetworkModifier
from nn_verification_visualisation.model.data.network_verification_config import NetworkVerificationConfig
from nn_verification_visualisation.model.data_loader.neural_network_loader import NeuralNetworkLoader
from onnx import helper, TensorProto

def test_network_modifier():
    node_x = helper.make_node("Gemm",["input","W1", "B1"],"fc1_out","FullyConnected1")
    node_y = helper.make_node("Relu","fc1_out","relu_out","Relu1")
    node_z = helper.make_node("Gemm",["relu_out","W2","B2"],"output","FullyConnected2")
    tensor_1 = helper.make_tensor("W1",1,[4,8],[
        0.5830841660499573,
        0.13663217425346375,
        -0.8730549216270447,
        1.6359570026397705,
        0.5052473545074463,
        0.041094984859228134,
        1.3184300661087036,
        -0.46083319187164307,
        -2.108297109603882,
        -1.8322064876556396,
        0.6757072806358337,
        0.9984564781188965,
        0.10428928583860397,
        0.378872811794281,
        -0.02697055973112583,
        0.46254122257232666,
        -0.1807435154914856,
        0.48952049016952515,
        -0.497354656457901,
        -2.0889649391174316,
        -0.18687047064304352,
        -1.3561513423919678,
        -1.1919441223144531,
        0.8655736446380615,
        -0.008533981628715992,
        1.4420583248138428,
        -1.1563801765441895,
        -1.1114248037338257,
        1.4735842943191528,
        0.3826438784599304,
        0.620539665222168,
        -0.34044674038887024
    ])
    tensor_2 = helper.make_tensor("B1",1,[8],[
    -0.09428485482931137,
    0.2977374494075775,
    -0.8388001918792725,
    -1.853492021560669,
    0.030526340007781982,
    1.931421160697937,
    -0.6110124588012695,
    0.01669270358979702])
    tensor_3 = helper.make_tensor("W2",1,[8,2],[
        -0.4191172420978546,
        -0.05619022622704506,
        2.037233591079712,
        1.1506322622299194,
        -0.07131075114011765,
        -0.6936957240104675,
        1.0400233268737793,
        0.1764814704656601,
        0.8922547698020935,
        -1.2435221672058105,
        -1.0535038709640503,
        -0.08947000652551651,
        0.3997858464717865,
        0.05722717568278313,
        -0.6842482089996338,
        1.056631326675415])
    tensor_4 = helper.make_tensor("B2",1,[2],[2.03241515,-0.148244113])
    test_type_1 = helper.make_tensor_type_proto(1,[4])
    test_type_2 = helper.make_tensor_type_proto(1,[2])
    test_input = helper.make_value_info("input",test_type_1)
    test_output = helper.make_value_info("output",test_type_2)
    test_graph = helper.make_graph([node_x,node_y,node_z],"test_graph",[test_input],[test_output],[tensor_1,tensor_2,tensor_3,tensor_4])
    test_model = helper.make_model(test_graph,producer_name="test_model")
    test_modification(test_model)


def test_modification(test_model: ModelProto):
    '''

    :param test_model:
    :return:
    '''
    test_directions = AlgorithmExecutor.calculate_directions(AlgorithmExecutor(),32)
    modified_test_model = NetworkModifier.custom_output_layer(NetworkModifier(), test_model,[(2,1),(1,2)],test_directions)
    assert(modified_test_model.graph.initializer[0].dims[0] == 4)
    assert(modified_test_model.graph.initializer[0].dims[1] == 8)
    assert(modified_test_model.graph.initializer[1].dims[0] == 8)
    assert(modified_test_model.graph.initializer[2].dims[0] == 8)
    assert(modified_test_model.graph.initializer[2].dims[1] == 3)
    assert(modified_test_model.graph.initializer[3].dims[0] == 3)
    assert(modified_test_model.graph.initializer[4].dims[0] == 3)
    assert(modified_test_model.graph.initializer[4].dims[1] == 32)
    assert(modified_test_model.graph.initializer[5].dims[0] == 32)
    assert(modified_test_model.graph.initializer[0].float_data == test_model.graph.initializer[0].float_data)
    assert (modified_test_model.graph.initializer[1].float_data == test_model.graph.initializer[1].float_data)
    assert (modified_test_model.graph.initializer[2].float_data == [
    [
        -0.4191172420978546,
        -0.05619022622704506,
        0
    ],
    [
        2.037233591079712,
        1.1506322622299194,
        0
    ],
    [
        -0.07131075114011765,
        -0.6936957240104675,
        1
    ],
    [
        1.0400233268737793,
        0.1764814704656601,
        0
    ],
    [
        0.8922547698020935,
        -1.2435221672058105,
        0
    ],
    [
        -1.0535038709640503,
        -0.08947000652551651,
        0
    ],
    [
        0.3997858464717865,
        0.05722717568278313,
        0
    ],
    [
        -0.6842482089996338,
        1.056631326675415,
        0
    ]
]
)
    assert (modified_test_model.graph.initializer[3].float_data == [
            2.0324151515960693,
            -0.14824411273002625,
            0
])


