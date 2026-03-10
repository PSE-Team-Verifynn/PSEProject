ALGORITHM_NAME = "Box IBP (NumPy)"
IS_DETERMINISTIC = True

import numpy as np
from onnx import numpy_helper


def _initializer_map(onnx_model) -> dict[str, np.ndarray]:
    return {
        initializer.name: np.asarray(numpy_helper.to_array(initializer), dtype=np.float64)
        for initializer in onnx_model.graph.initializer
    }


def _apply_gemm(
    lower: np.ndarray,
    upper: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    weight_pos = np.maximum(weight, 0.0)
    weight_neg = np.minimum(weight, 0.0)

    out_lower = lower @ weight_pos + upper @ weight_neg
    out_upper = upper @ weight_pos + lower @ weight_neg

    if bias is not None:
        out_lower = out_lower + bias
        out_upper = out_upper + bias

    return out_lower, out_upper


def calculate_output_bounds(onnx_model, input_bounds: np.ndarray) -> np.ndarray:
    """
    NumPy-only box interval propagation for feedforward ONNX models made from:
    - Gemm
    - Relu

    This is intended to work for TestFiles/NN1.onnx and similar MLP-style models.
    """
    if input_bounds.ndim != 2 or input_bounds.shape[1] != 2:
        raise ValueError("input_bounds must have shape (N, 2).")

    initializers = _initializer_map(onnx_model)
    lower_bounds: dict[str, np.ndarray] = {
        onnx_model.graph.input[0].name: input_bounds[:, 0].astype(np.float64, copy=True).reshape(1, -1)
    }
    upper_bounds: dict[str, np.ndarray] = {
        onnx_model.graph.input[0].name: input_bounds[:, 1].astype(np.float64, copy=True).reshape(1, -1)
    }

    for node in onnx_model.graph.node:
        if node.op_type == "Gemm":
            if len(node.input) < 2:
                raise ValueError(f"Gemm node {node.name!r} is missing inputs.")

            inp_name = node.input[0]
            weight_name = node.input[1]
            bias_name = node.input[2] if len(node.input) > 2 and node.input[2] else None

            if inp_name not in lower_bounds or inp_name not in upper_bounds:
                raise ValueError(f"Missing interval for input tensor {inp_name!r}.")
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
                bias = beta * initializers[bias_name].reshape(1, -1)

            out_lower, out_upper = _apply_gemm(lower_bounds[inp_name], upper_bounds[inp_name], weight, bias)
        elif node.op_type == "Relu":
            inp_name = node.input[0]
            if inp_name not in lower_bounds or inp_name not in upper_bounds:
                raise ValueError(f"Missing interval for input tensor {inp_name!r}.")
            out_lower = np.maximum(lower_bounds[inp_name], 0.0)
            out_upper = np.maximum(upper_bounds[inp_name], 0.0)
        else:
            raise ValueError(
                f"Unsupported ONNX operator {node.op_type!r} in node {node.name!r}. "
                "This NumPy box algorithm currently supports Gemm and Relu only."
            )

        output_name = node.output[0]
        lower_bounds[output_name] = out_lower
        upper_bounds[output_name] = out_upper

    output_name = onnx_model.graph.output[0].name
    if output_name not in lower_bounds or output_name not in upper_bounds:
        raise ValueError(f"Could not compute bounds for output tensor {output_name!r}.")

    return np.stack(
        [
            lower_bounds[output_name].reshape(-1),
            upper_bounds[output_name].reshape(-1),
        ],
        axis=1,
    )
