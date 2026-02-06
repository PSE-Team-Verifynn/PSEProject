ALGORITHM_NAME = "Sampling Baseline (ONNXRuntime)"
IS_DETERMINISTIC = True

import numpy as np
import onnxruntime as ort


def _reshape_samples(samples: np.ndarray, input_shape: list, input_name: str) -> np.ndarray:
    if not input_shape or len(input_shape) <= 2:
        return samples

    tail = input_shape[1:]
    if any(dim is None for dim in tail):
        raise RuntimeError(
            "Input rank > 2 has dynamic dimensions; cannot infer sample reshape."
        )

    expected = int(np.prod(tail))
    got = samples.shape[1]
    if expected != got:
        raise RuntimeError(
            f"Input '{input_name}' expects {expected} features but bounds provide {got}."
        )
    return samples.reshape((samples.shape[0], *tail))


def calculate_output_bounds(onnx_model, input_bounds: np.ndarray) -> np.ndarray:
    if input_bounds.ndim != 2 or input_bounds.shape[1] != 2:
        raise ValueError("input_bounds must have shape (N, 2).")

    num_samples = 3000
    rng = np.random.default_rng(0)

    lo = input_bounds[:, 0].astype(np.float32)
    hi = input_bounds[:, 1].astype(np.float32)
    samples = rng.uniform(low=lo, high=hi, size=(num_samples, input_bounds.shape[0])).astype(np.float32)

    session = ort.InferenceSession(onnx_model.SerializeToString(), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    if not inputs:
        raise RuntimeError("Model has no inputs.")
    input_name = inputs[0].name
    input_shape = inputs[0].shape
    samples = _reshape_samples(samples, input_shape, input_name)

    output_name = session.get_outputs()[0].name
    first_dim = input_shape[0] if input_shape else None

    if first_dim == 1:
        outputs = []
        for i in range(samples.shape[0]):
            out = session.run([output_name], {input_name: samples[i:i + 1]})[0]
            outputs.append(out)
        output = np.concatenate(outputs, axis=0)
    else:
        output = session.run([output_name], {input_name: samples})[0]

    output = output.reshape(output.shape[0], -1)
    lower = np.min(output, axis=0)
    upper = np.max(output, axis=0)
    return np.stack([lower, upper], axis=1)
