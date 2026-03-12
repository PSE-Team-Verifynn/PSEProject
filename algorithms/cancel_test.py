ALGORITHM_NAME = "Cancel Test"
IS_DETERMINISTIC = True

import time

import numpy as np


def _infer_output_size(onnx_model) -> int:
    if onnx_model.graph.output:
        tensor_type = onnx_model.graph.output[0].type.tensor_type
        dims = tensor_type.shape.dim
        if len(dims) >= 2 and dims[1].dim_value > 0:
            return int(dims[1].dim_value)
        if len(dims) == 1 and dims[0].dim_value > 0:
            return int(dims[0].dim_value)

    for init in onnx_model.graph.initializer:
        if init.name == "output_initializer_B" and len(init.dims) == 1 and init.dims[0] > 0:
            return int(init.dims[0])

    raise ValueError("Could not infer output size from model.")


def calculate_output_bounds(onnx_model, input_bounds: np.ndarray) -> np.ndarray:
    if input_bounds.ndim != 2 or input_bounds.shape[1] != 2:
        raise ValueError("input_bounds must have shape (N, 2).")

    time.sleep(10)

    output_size = _infer_output_size(onnx_model)
    lower = np.full(output_size, -1.0, dtype=float)
    upper = np.full(output_size, 1.0, dtype=float)
    return np.stack([lower, upper], axis=1)
