ALGORITHM_NAME = "Interval Width Baseline (No Runtime)"
IS_DETERMINISTIC = True

import numpy as np


def _infer_output_size(onnx_model) -> int:
    # Prefer declared graph output shape.
    if onnx_model.graph.output:
        tensor_type = onnx_model.graph.output[0].type.tensor_type
        dims = tensor_type.shape.dim
        if len(dims) >= 2 and dims[1].dim_value > 0:
            return int(dims[1].dim_value)
        if len(dims) == 1 and dims[0].dim_value > 0:
            return int(dims[0].dim_value)

    # Fallback for models modified by custom_output_layer.
    for init in onnx_model.graph.initializer:
        if init.name == "output_initializer_B" and len(init.dims) == 1 and init.dims[0] > 0:
            return int(init.dims[0])

    raise ValueError("Could not infer output size from model.")


def calculate_output_bounds(onnx_model, input_bounds: np.ndarray) -> np.ndarray:
    """
    Lightweight deterministic baseline with no runtime backend.
    It produces symmetric output bounds based on aggregate input interval width.

    This is not a sound verifier for real model behavior, but is useful to test:
    - algorithm loading
    - execution flow
    - plotting pipeline
    """
    if input_bounds.ndim != 2 or input_bounds.shape[1] != 2:
        raise ValueError("input_bounds must have shape (N, 2).")

    output_size = _infer_output_size(onnx_model)

    lower = input_bounds[:, 0].astype(float)
    upper = input_bounds[:, 1].astype(float)
    half_widths = np.maximum(0.0, (upper - lower) * 0.5)

    # Single deterministic scale for all outputs, with tiny per-output variation
    # so polygons are visible and not perfectly identical.
    base_radius = float(np.sum(half_widths))
    radii = np.array([base_radius * (1.0 + 0.02 * i) for i in range(output_size)], dtype=float)

    out_lower = -radii
    out_upper = radii
    return np.stack([out_lower, out_upper], axis=1)
