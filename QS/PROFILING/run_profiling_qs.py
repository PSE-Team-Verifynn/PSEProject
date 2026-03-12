from __future__ import annotations

import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from nn_verification_visualisation.controller.process_manager.algorithm_executor import AlgorithmExecutor
from nn_verification_visualisation.model.data.network_verification_config import NetworkVerificationConfig
from nn_verification_visualisation.model.data.storage import Storage
from nn_verification_visualisation.model.data_loader.neural_network_loader import NeuralNetworkLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qs_common import PROFILING_OUT_DIR, ROOT, ensure_output_dir, input_dim_from_model, maxrss_kb, write_csv, write_json, write_profiling_plot


def load_network_and_default_bounds(network_path: Path):
    nn_res = NeuralNetworkLoader().load_neural_network(str(network_path))
    if not nn_res.is_success:
        raise nn_res.error

    network = nn_res.data
    input_dim = input_dim_from_model(network_path)
    _config = NetworkVerificationConfig(network, [input_dim])
    bounds_np = np.array([(0.0, 1.0) for _ in range(input_dim)], dtype=float)
    return network, bounds_np


def profile_case_worker(network_path: str, algorithm_path: str, selected_neurons: list[tuple[int, int]], queue) -> None:
    try:
        Storage().num_directions = 16
        network, bounds_np = load_network_and_default_bounds(Path(network_path))
        start_time = time.perf_counter()
        result = AlgorithmExecutor().execute_algorithm(network.model, bounds_np, algorithm_path, selected_neurons)
        runtime_ms = (time.perf_counter() - start_time) * 1000
        if not result.is_success:
            raise result.error
        queue.put({"runtime_ms": round(runtime_ms, 3), "maxrss_kb": maxrss_kb()})
    except BaseException as error:
        queue.put({"error": repr(error)})


def run_profiling_checks() -> list[dict]:
    cases = [
        {
            "case": "simple_3_layer_x10",
            "network": ROOT / "Files" / "simple_3_layer_net x10.onnx",
            "algorithm": ROOT / "algorithms" / "box_ibp_numpy.py",
            "selected_neurons": [(0, 0), (0, 1)],
        },
        {
            "case": "simple_3_layer_x50",
            "network": ROOT / "Files" / "simple_3_layer_net x50.onnx",
            "algorithm": ROOT / "algorithms" / "box_ibp_numpy.py",
            "selected_neurons": [(0, 0), (0, 1)],
        },
        {
            "case": "simple_3_layer_x100",
            "network": ROOT / "Files" / "simple_3_layer_net x100.onnx",
            "algorithm": ROOT / "algorithms" / "box_ibp_numpy.py",
            "selected_neurons": [(0, 0), (0, 1)],
        },
        {
            "case": "NN2",
            "network": ROOT / "Files" / "NN2.onnx",
            "algorithm": ROOT / "algorithms" / "box_ibp_numpy.py",
            "selected_neurons": [(0, 99), (0, 199)],
        },
        {
            "case": "NN1_small",
            "network": ROOT / "Files" / "NN1.onnx",
            "algorithm": ROOT / "algorithms" / "box_ibp_numpy.py",
            "selected_neurons": [(0, 0), (0, 1)],
        },
    ]

    ctx = get_context("spawn")
    results = []
    for case in cases:
        queue = ctx.Queue()
        process = ctx.Process(
            target=profile_case_worker,
            args=(str(case["network"]), str(case["algorithm"]), case["selected_neurons"], queue),
        )
        process.start()
        process.join()
        payload = queue.get()
        if "error" in payload:
            raise RuntimeError(f"{case['case']} failed: {payload['error']}")
        results.append(
            {
                "case": case["case"],
                "runtime_ms": payload["runtime_ms"],
                "maxrss_kb": payload["maxrss_kb"],
            }
        )

    return results


def main() -> None:
    ensure_output_dir(PROFILING_OUT_DIR)
    rows = run_profiling_checks()
    write_json(PROFILING_OUT_DIR / "profiling_results.json", rows)
    write_csv(PROFILING_OUT_DIR / "profiling_metrics.csv", rows, ["case", "runtime_ms", "maxrss_kb"])
    write_profiling_plot(rows)
    print({"profiling_cases": len(rows)})


if __name__ == "__main__":
    main()
