from __future__ import annotations

from logging import Logger
from typing import Iterable

import numpy as np
import onnx
import onnxruntime as ort

from nn_verification_visualisation.model.data.neural_network import NeuralNetwork
from nn_verification_visualisation.controller.process_manager.sample_metric_registry import get_metric_map
from nn_verification_visualisation.controller.process_manager.network_modifier import NetworkModifier


def run_samples_for_bounds(
    network: NeuralNetwork,
    bounds: list[tuple[float, float]],
    num_samples: int,
    metrics: Iterable[str],
) -> dict:
    logger = Logger(__name__)
    if num_samples <= 0:
        logger.error("num_samples must be positive")
        raise ValueError("num_samples must be positive")


    metric_map = get_metric_map()
    metric_list = [metric for metric in metrics if metric in metric_map]
    if not metric_list:
        logger.error("No valid metrics selected")
        raise ValueError("No valid metrics selected")

    model = onnx.load(network.path)
    model = NetworkModifier.with_all_outputs(model)
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    if not inputs:
        logger.error("Model has no inputs")
        raise RuntimeError("Model has no inputs")
    input_name = inputs[0].name

    output_names: list[str] = []
    for output in model.graph.output:
        if output.name:
            output_names.append(output.name)
    if not output_names:
        logger.error("Model has no outputs to sample")
        raise RuntimeError("Model has no outputs to sample")

    low = np.array([pair[0] for pair in bounds], dtype=np.float32)
    high = np.array([pair[1] for pair in bounds], dtype=np.float32)
    samples = np.random.uniform(low=low, high=high, size=(num_samples, len(bounds))).astype(np.float32)
    input_shape = inputs[0].shape
    first_dim = input_shape[0] if input_shape else None
    if input_shape and len(input_shape) > 2:
        # Reshape flat samples to match expected input rank (excluding batch).
        expected_rank = len(input_shape)
        expected_tail = input_shape[1:]
        total_features = samples.shape[1]
        if not all(dim is not None for dim in expected_tail):
            logger.error("Sample input rank mismatch. "
                f"Input '{input_name}' expects rank {expected_rank} shape {input_shape}, "
                "but its dimensions are dynamic. Use bounds that match the model input shape.")
            raise RuntimeError(
                "Sample input rank mismatch. "
                f"Input '{input_name}' expects rank {expected_rank} shape {input_shape}, "
                "but its dimensions are dynamic. Use bounds that match the model input shape."
            )
        expected_size = int(np.prod(expected_tail))
        if expected_size != total_features:
            logger.error("Sample input size mismatch. "
                f"Input '{input_name}' expects shape {input_shape} "
                f"(size {expected_size}), but bounds provide {total_features} values.")
            raise RuntimeError(
                "Sample input size mismatch. "
                f"Input '{input_name}' expects shape {input_shape} "
                f"(size {expected_size}), but bounds provide {total_features} values."
            )
        samples = samples.reshape((num_samples, *expected_tail))
    if first_dim == 1:
        out_lists = [[] for _ in output_names]
        for i in range(samples.shape[0]):
            out = session.run(output_names, {input_name: samples[i:i + 1]})
            if not out:
                logger.error("Model produced no outputs")
                raise RuntimeError("Model produced no outputs")
            for idx, item in enumerate(out):
                out_lists[idx].append(item)
        outputs = [np.concatenate(items, axis=0) for items in out_lists]
    else:
        outputs = session.run(output_names, {input_name: samples})
        if not outputs:
            logger.error("Model produced no outputs")
            raise RuntimeError("Model produced no outputs")

    output_entries: list[dict] = []
    for name, output in zip(output_names, outputs):
        if output.ndim == 1:
            output = output.reshape((num_samples, 1))
        metric_values: dict[str, list[float]] = {}
        for metric_key in metric_list:
            metric = metric_map[metric_key]
            value = metric.compute(output)
            metric_values[metric_key] = value.reshape(-1).astype(float).tolist()
        output_entries.append(
            {
                "name": name,
                "shape": list(output.shape[1:]),
                "values": metric_values,
            }
        )

    return {
        "num_samples": num_samples,
        "metrics": metric_list,
        "outputs": output_entries,
    }
