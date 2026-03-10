ALGORITHM_NAME = "Simple Zonotope"
IS_DETERMINISTIC = True

import numpy as np
from onnx import numpy_helper


def _initializer_map(onnx_model) -> dict[str, np.ndarray]:
    return {
        initializer.name: np.asarray(numpy_helper.to_array(initializer), dtype=np.float64)
        for initializer in onnx_model.graph.initializer
    }


def _input_zonotope(input_bounds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lower = input_bounds[:, 0].astype(np.float64, copy=False)
    upper = input_bounds[:, 1].astype(np.float64, copy=False)
    center = (lower + upper) * 0.5
    radius = np.maximum(0.0, (upper - lower) * 0.5)
    generators = np.diag(radius)
    return center, generators


def _zonotope_interval(center: np.ndarray, generators: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radius = np.sum(np.abs(generators), axis=0) if generators.size else np.zeros_like(center)
    return center - radius, center + radius


def _apply_gemm(
    center: np.ndarray,
    generators: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    out_center = center @ weight
    if bias is not None:
        out_center = out_center + bias.reshape(-1)

    if generators.size == 0:
        out_generators = np.zeros((0, weight.shape[1]), dtype=np.float64)
    else:
        out_generators = generators @ weight
    return out_center, out_generators


def _apply_relu(center: np.ndarray, generators: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = _zonotope_interval(center, generators)

    stable_positive = lower >= 0.0
    stable_negative = upper <= 0.0
    unstable = ~(stable_positive | stable_negative)

    out_center = center.copy()
    out_generators = generators.copy()

    if np.any(stable_negative):
        out_center[stable_negative] = 0.0
        if out_generators.size:
            out_generators[:, stable_negative] = 0.0

    if np.any(unstable):
        # For unstable ReLUs, fall back to the interval hull [0, upper].
        # This is sound and keeps the implementation simple, at the cost of precision.
        out_center[unstable] = upper[unstable] * 0.5
        if out_generators.size:
            out_generators[:, unstable] = 0.0

        new_generators = np.zeros((int(np.sum(unstable)), center.shape[0]), dtype=np.float64)
        unstable_indices = np.flatnonzero(unstable)
        new_generators[np.arange(len(unstable_indices)), unstable_indices] = upper[unstable] * 0.5
        out_generators = np.vstack([out_generators, new_generators]) if out_generators.size else new_generators

    return out_center, out_generators


def calculate_output_bounds(onnx_model, input_bounds: np.ndarray) -> np.ndarray:
    """
    Sound zonotope-style propagation for feedforward ONNX models made from:
    - Gemm
    - Relu

    Linear layers preserve the full zonotope.
    Unstable ReLUs are overapproximated by replacing the affected coordinate with
    an independent interval hull [0, upper], which is sound but coarse.
    """
    if input_bounds.ndim != 2 or input_bounds.shape[1] != 2:
        raise ValueError("input_bounds must have shape (N, 2).")

    initializers = _initializer_map(onnx_model)
    tensor_state: dict[str, tuple[np.ndarray, np.ndarray]] = {
        onnx_model.graph.input[0].name: _input_zonotope(input_bounds)
    }

    for node in onnx_model.graph.node:
        if node.op_type == "Gemm":
            if len(node.input) < 2:
                raise ValueError(f"Gemm node {node.name!r} is missing inputs.")

            inp_name = node.input[0]
            weight_name = node.input[1]
            bias_name = node.input[2] if len(node.input) > 2 and node.input[2] else None

            if inp_name not in tensor_state:
                raise ValueError(f"Missing zonotope for input tensor {inp_name!r}.")
            if weight_name not in initializers:
                raise ValueError(f"Gemm weight initializer {weight_name!r} was not found.")

            trans_a = 0
            trans_b = 0
            alpha = 1.0
            beta = 1.0
            for attribute in node.attribute:
                if attribute.name == "transA":
                    trans_a = attribute.i
                elif attribute.name == "transB":
                    trans_b = attribute.i
                elif attribute.name == "alpha":
                    alpha = attribute.f
                elif attribute.name == "beta":
                    beta = attribute.f

            if trans_a:
                raise ValueError(f"Gemm node {node.name!r} uses transA=1, which is not supported.")

            weight = initializers[weight_name]
            if trans_b:
                weight = weight.T
            weight = alpha * weight

            bias = None
            if bias_name is not None:
                if bias_name not in initializers:
                    raise ValueError(f"Gemm bias initializer {bias_name!r} was not found.")
                bias = beta * initializers[bias_name]

            out_state = _apply_gemm(*tensor_state[inp_name], weight, bias)
        elif node.op_type == "Relu":
            inp_name = node.input[0]
            if inp_name not in tensor_state:
                raise ValueError(f"Missing zonotope for input tensor {inp_name!r}.")
            out_state = _apply_relu(*tensor_state[inp_name])
        else:
            raise ValueError(
                f"Unsupported ONNX operator {node.op_type!r} in node {node.name!r}. "
                "This simple zonotope algorithm currently supports Gemm and Relu only."
            )

        tensor_state[node.output[0]] = out_state

    output_name = onnx_model.graph.output[0].name
    if output_name not in tensor_state:
        raise ValueError(f"Could not compute bounds for output tensor {output_name!r}.")

    lower, upper = _zonotope_interval(*tensor_state[output_name])
    return np.stack([lower.reshape(-1), upper.reshape(-1)], axis=1)
