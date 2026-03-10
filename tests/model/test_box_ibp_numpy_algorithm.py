from itertools import product
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

from nn_verification_visualisation.model.data_loader.algorithm_loader import AlgorithmLoader


def _evaluate_nn1_corner(model: onnx.ModelProto, point: np.ndarray) -> np.ndarray:
    values = {initializer.name: np.asarray(numpy_helper.to_array(initializer), dtype=np.float64) for initializer in model.graph.initializer}
    current = point.reshape(1, -1)

    for node in model.graph.node:
        if node.op_type == "Gemm":
            current = current @ values[node.input[1]] + values[node.input[2]].reshape(1, -1)
        elif node.op_type == "Relu":
            current = np.maximum(current, 0.0)
        else:
            raise AssertionError(f"Unexpected operator in NN1 fixture: {node.op_type}")

    return current.reshape(-1)


def test_box_ibp_numpy_is_sound_for_nn1():
    repo_root = Path(__file__).resolve().parents[2]
    model = onnx.load(repo_root / "TestFiles" / "NN1.onnx")

    input_dim = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
    input_bounds = np.column_stack([np.full(input_dim, -0.5), np.full(input_dim, 0.75)])

    algo_res = AlgorithmLoader().load_calculate_output_bounds(str(repo_root / "algorithms" / "box_ibp_numpy.py"))
    assert algo_res.is_success, algo_res.error

    interval_bounds = algo_res.data(model, input_bounds)
    assert interval_bounds.shape[1] == 2

    corner_outputs = []
    for selector in product([0, 1], repeat=input_dim):
        point = np.array([input_bounds[i, selector[i]] for i in range(input_dim)], dtype=np.float64)
        corner_outputs.append(_evaluate_nn1_corner(model, point))

    corner_outputs = np.asarray(corner_outputs)
    exact_bounds = np.stack([corner_outputs.min(axis=0), corner_outputs.max(axis=0)], axis=1)

    assert np.all(interval_bounds[:, 0] <= exact_bounds[:, 0] + 1e-9)
    assert np.all(interval_bounds[:, 1] >= exact_bounds[:, 1] - 1e-9)
