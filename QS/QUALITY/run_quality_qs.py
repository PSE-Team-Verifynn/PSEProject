from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from nn_verification_visualisation.controller.process_manager.algorithm_executor import AlgorithmExecutor
from nn_verification_visualisation.controller.process_manager.network_modifier import NetworkModifier
from nn_verification_visualisation.controller.process_manager.sample_runner import run_samples_for_bounds
from nn_verification_visualisation.model.data.network_verification_config import NetworkVerificationConfig
from nn_verification_visualisation.model.data.storage import Storage
from nn_verification_visualisation.model.data_loader.input_bounds_loader import InputBoundsLoader
from nn_verification_visualisation.model.data_loader.neural_network_loader import NeuralNetworkLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qs_common import QUALITY_OUT_DIR, ROOT, ensure_output_dir, input_dim_from_model, maxrss_kb, run_model_samples, write_csv, write_json, write_quality_plot


def load_network_and_bounds(network_path, bounds_path):
    nn_res = NeuralNetworkLoader().load_neural_network(str(network_path))
    if not nn_res.is_success:
        raise nn_res.error

    network = nn_res.data
    input_dim = input_dim_from_model(network_path)
    config = NetworkVerificationConfig(network, [input_dim])
    bounds_res = InputBoundsLoader().load_input_bounds(str(bounds_path), config)
    if not bounds_res.is_success:
        raise bounds_res.error

    bounds_map = bounds_res.data
    bounds_np = np.array([bounds_map[index] for index in range(len(bounds_map))], dtype=float)
    return network, bounds_map, bounds_np


def run_quality_checks() -> list[dict]:
    Storage().num_directions = 16
    np.random.seed(7)
    random.seed(7)

    cases = [
        {
            "case": "NN1_BoxIBP",
            "network": ROOT / "Files" / "NN1.onnx",
            "bounds": ROOT / "Files" / "B1.csv",
            "algorithm": ROOT / "algorithms" / "box_ibp_numpy.py",
            "selected_neurons": [(0, 0), (0, 1)],
            "samples": 2000,
        },
        {
            "case": "NN1_Zonotope",
            "network": ROOT / "Files" / "NN1.onnx",
            "bounds": ROOT / "Files" / "B1.csv",
            "algorithm": ROOT / "algorithms" / "simple_zonotope.py",
            "selected_neurons": [(0, 0), (0, 1)],
            "samples": 2000,
        },
        {
            "case": "NN2_BoxIBP",
            "network": ROOT / "Files" / "NN2.onnx",
            "bounds": ROOT / "Files" / "B1.csv",
            "algorithm": ROOT / "algorithms" / "box_ibp_numpy.py",
            "selected_neurons": [(0, 99), (0, 199)],
            "samples": 1000,
        },
        {
            "case": "NN2_Zonotope",
            "network": ROOT / "Files" / "NN2.onnx",
            "bounds": ROOT / "Files" / "B1.csv",
            "algorithm": ROOT / "algorithms" / "simple_zonotope.py",
            "selected_neurons": [(0, 99), (0, 199)],
            "samples": 1000,
        },
    ]

    results = []
    for case in cases:
        network, bounds_map, bounds_np = load_network_and_bounds(case["network"], case["bounds"])

        start_rss = maxrss_kb()
        start_time = time.perf_counter()
        result = AlgorithmExecutor().execute_algorithm(network.model, bounds_np, str(case["algorithm"]), case["selected_neurons"])
        runtime_ms = (time.perf_counter() - start_time) * 1000
        memory_delta_kb = max(0, maxrss_kb() - start_rss)
        if not result.is_success:
            raise result.error

        computed_bounds, directions = result.data
        low = np.array([pair[0] for pair in bounds_map.values()], dtype=np.float32)
        high = np.array([pair[1] for pair in bounds_map.values()], dtype=np.float32)
        samples = np.random.uniform(low=low, high=high, size=(case["samples"], len(low))).astype(np.float32)

        modified_model = NetworkModifier().custom_output_layer(network.model, case["selected_neurons"], directions)
        session = ort.InferenceSession(modified_model.SerializeToString(), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        actual_outputs = run_model_samples(session, input_name, output_name, samples)

        actual_min = actual_outputs.min(axis=0)
        actual_max = actual_outputs.max(axis=0)
        computed_min = computed_bounds[:, 0]
        computed_max = computed_bounds[:, 1]
        contained = (actual_min >= computed_min - 1e-6) & (actual_max <= computed_max + 1e-6)
        width = computed_max - computed_min
        actual_width = actual_max - actual_min
        tightness = np.divide(actual_width, width, out=np.zeros_like(actual_width), where=width > 0)

        sample_summary = run_samples_for_bounds(network, list(bounds_map.values()), min(case["samples"], 1000), ["max", "mean", "range"])

        results.append(
            {
                "case": case["case"],
                "runtime_ms": round(runtime_ms, 3),
                "memory_delta_kb": int(memory_delta_kb),
                "containment_pass": bool(contained.all()),
                "containment_ratio": round(float(contained.mean()), 4),
                "avg_bound_width": round(float(width.mean()), 6),
                "avg_sample_width": round(float(actual_width.mean()), 6),
                "avg_tightness_ratio": round(float(tightness.mean()), 6),
                "sample_metric_outputs": len(sample_summary["outputs"]),
                "bounded_directions": int(contained.sum()),
                "checked_directions": int(len(contained)),
            }
        )

    return results


def main() -> None:
    ensure_output_dir(QUALITY_OUT_DIR)
    rows = run_quality_checks()
    write_json(QUALITY_OUT_DIR / "quality_results.json", rows)
    write_csv(
        QUALITY_OUT_DIR / "quality_metrics.csv",
        rows,
        ["case", "runtime_ms", "memory_delta_kb", "containment_pass", "containment_ratio", "avg_bound_width", "avg_sample_width", "avg_tightness_ratio", "sample_metric_outputs", "bounded_directions", "checked_directions"],
    )
    write_quality_plot(rows)
    print({"quality_cases": len(rows)})


if __name__ == "__main__":
    main()
