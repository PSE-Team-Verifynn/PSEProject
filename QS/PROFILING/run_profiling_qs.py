from __future__ import annotations

import statistics
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

from qs_common import (
    PROFILING_OUT_DIR,
    ROOT,
    ensure_output_dir,
    maxrss_kb,
    write_csv,
    write_json,
    write_profiling_plot,
)

REPEATS = 5


def load_network_and_default_bounds(network_path: Path):
    nn_res = NeuralNetworkLoader().load_neural_network(str(network_path))
    if not nn_res.is_success:
        raise nn_res.error

    network = nn_res.data
    input_dim = network.model.graph.input[0].type.tensor_type.shape.dim[-1].dim_value
    if not input_dim:
        raise ValueError(f"Could not determine input dimension for {network_path}")
    _config = NetworkVerificationConfig(network, [input_dim])
    bounds_np = np.array([(0.0, 1.0) for _ in range(input_dim)], dtype=float)
    return network, bounds_np


def profile_load_worker(network_path: str, queue) -> None:
    try:
        Storage().num_directions = 16
        baseline_rss_kb = maxrss_kb()
        load_start = time.perf_counter()
        load_network_and_default_bounds(Path(network_path))
        load_runtime_ms = (time.perf_counter() - load_start) * 1000
        loaded_peak_rss_kb = maxrss_kb()
        queue.put(
            {
                "load_runtime_ms": round(load_runtime_ms, 3),
                "baseline_rss_kb": baseline_rss_kb,
                "loaded_peak_rss_kb": loaded_peak_rss_kb,
                "load_memory_kb": max(loaded_peak_rss_kb - baseline_rss_kb, 0),
            }
        )
    except BaseException as error:
        queue.put({"error": repr(error)})


def profile_algorithm_worker(network_path: str, algorithm_path: str, selected_neurons: list[tuple[int, int]], queue) -> None:
    try:
        Storage().num_directions = 16
        network, bounds_np = load_network_and_default_bounds(Path(network_path))
        loaded_peak_rss_kb = maxrss_kb()

        algorithm_start = time.perf_counter()
        result = AlgorithmExecutor().execute_algorithm(
            network.model,
            bounds_np,
            algorithm_path,
            selected_neurons,
            Storage().num_directions,
        )
        algorithm_runtime_ms = (time.perf_counter() - algorithm_start) * 1000
        if not result.is_success:
            raise result.error

        peak_rss_kb = maxrss_kb()
        queue.put(
            {
                "algorithm_runtime_ms": round(algorithm_runtime_ms, 3),
                "algorithm_memory_kb": max(peak_rss_kb - loaded_peak_rss_kb, 0),
                "peak_rss_kb": peak_rss_kb,
                "loaded_peak_rss_kb": loaded_peak_rss_kb,
            }
        )
    except BaseException as error:
        queue.put({"error": repr(error)})


def _run_case_worker(ctx, target, args: tuple) -> dict:
    queue = ctx.Queue()
    process = ctx.Process(target=target, args=(*args, queue))
    process.start()
    process.join()
    payload = queue.get()
    if "error" in payload:
        raise RuntimeError(payload["error"])
    return payload


def _summarize_numeric(values: list[float | int], digits: int = 3) -> tuple[float, float, float]:
    median_value = round(float(statistics.median(values)), digits)
    min_value = round(float(min(values)), digits)
    max_value = round(float(max(values)), digits)
    return median_value, min_value, max_value


def run_profiling_checks() -> list[dict]:
    cases = [
        {
            "case": "simple_3_layer_x10",
            "network": ROOT / "TestFiles" / "simple_3_layer_net x10.onnx",
            "algorithm": ROOT / "algorithms" / "box_ibp_numpy.py",
            "selected_neurons": [(0, 0), (0, 1)],
        },
        {
            "case": "simple_3_layer_x50",
            "network": ROOT / "TestFiles" / "simple_3_layer_net x50.onnx",
            "algorithm": ROOT / "algorithms" / "box_ibp_numpy.py",
            "selected_neurons": [(0, 0), (0, 1)],
        },
        {
            "case": "simple_3_layer_x100",
            "network": ROOT / "TestFiles" / "simple_3_layer_net x100.onnx",
            "algorithm": ROOT / "algorithms" / "box_ibp_numpy.py",
            "selected_neurons": [(0, 0), (0, 1)],
        },
        {
            "case": "NN1_small",
            "network": ROOT / "TestFiles" / "NN1.onnx",
            "algorithm": ROOT / "algorithms" / "box_ibp_numpy.py",
            "selected_neurons": [(0, 0), (0, 1)],
        },
    ]

    ctx = get_context("spawn")
    results = []
    for case in cases:
        load_runs = [_run_case_worker(ctx, profile_load_worker, (str(case["network"]),)) for _ in range(REPEATS)]
        algorithm_runs = [
            _run_case_worker(
                ctx,
                profile_algorithm_worker,
                (str(case["network"]), str(case["algorithm"]), case["selected_neurons"]),
            )
            for _ in range(REPEATS)
        ]
        load_runtime_ms, load_runtime_min_ms, load_runtime_max_ms = _summarize_numeric(
            [run["load_runtime_ms"] for run in load_runs]
        )
        algorithm_runtime_ms, algorithm_runtime_min_ms, algorithm_runtime_max_ms = _summarize_numeric(
            [run["algorithm_runtime_ms"] for run in algorithm_runs]
        )
        baseline_rss_kb, baseline_rss_min_kb, baseline_rss_max_kb = _summarize_numeric(
            [run["baseline_rss_kb"] for run in load_runs],
            digits=0,
        )
        loaded_rss_kb, loaded_rss_min_kb, loaded_rss_max_kb = _summarize_numeric(
            [run["loaded_peak_rss_kb"] for run in load_runs],
            digits=0,
        )
        load_memory_kb, load_memory_min_kb, load_memory_max_kb = _summarize_numeric(
            [run["load_memory_kb"] for run in load_runs],
            digits=0,
        )
        algorithm_memory_kb, algorithm_memory_min_kb, algorithm_memory_max_kb = _summarize_numeric(
            [run["algorithm_memory_kb"] for run in algorithm_runs],
            digits=0,
        )
        peak_rss_kb, peak_rss_min_kb, peak_rss_max_kb = _summarize_numeric(
            [run["peak_rss_kb"] for run in algorithm_runs],
            digits=0,
        )
        results.append(
            {
                "case": case["case"],
                "repeats": REPEATS,
                "load_runtime_ms": load_runtime_ms,
                "load_runtime_min_ms": load_runtime_min_ms,
                "load_runtime_max_ms": load_runtime_max_ms,
                "algorithm_runtime_ms": algorithm_runtime_ms,
                "algorithm_runtime_min_ms": algorithm_runtime_min_ms,
                "algorithm_runtime_max_ms": algorithm_runtime_max_ms,
                "runtime_ms": round(load_runtime_ms + algorithm_runtime_ms, 3),
                "baseline_rss_kb": int(baseline_rss_kb),
                "baseline_rss_min_kb": int(baseline_rss_min_kb),
                "baseline_rss_max_kb": int(baseline_rss_max_kb),
                "loaded_rss_kb": int(loaded_rss_kb),
                "loaded_rss_min_kb": int(loaded_rss_min_kb),
                "loaded_rss_max_kb": int(loaded_rss_max_kb),
                "load_memory_kb": int(load_memory_kb),
                "load_memory_min_kb": int(load_memory_min_kb),
                "load_memory_max_kb": int(load_memory_max_kb),
                "algorithm_memory_kb": int(algorithm_memory_kb),
                "algorithm_memory_min_kb": int(algorithm_memory_min_kb),
                "algorithm_memory_max_kb": int(algorithm_memory_max_kb),
                "peak_rss_kb": int(peak_rss_kb),
                "peak_rss_min_kb": int(peak_rss_min_kb),
                "peak_rss_max_kb": int(peak_rss_max_kb),
                "maxrss_kb": int(peak_rss_kb),
            }
        )

    return results


def main() -> None:
    ensure_output_dir(PROFILING_OUT_DIR)
    rows = run_profiling_checks()
    write_json(PROFILING_OUT_DIR / "profiling_results.json", rows)
    write_csv(
        PROFILING_OUT_DIR / "profiling_metrics.csv",
        rows,
        [
            "case",
            "repeats",
            "load_runtime_ms",
            "load_runtime_min_ms",
            "load_runtime_max_ms",
            "algorithm_runtime_ms",
            "algorithm_runtime_min_ms",
            "algorithm_runtime_max_ms",
            "runtime_ms",
            "baseline_rss_kb",
            "baseline_rss_min_kb",
            "baseline_rss_max_kb",
            "loaded_rss_kb",
            "loaded_rss_min_kb",
            "loaded_rss_max_kb",
            "load_memory_kb",
            "load_memory_min_kb",
            "load_memory_max_kb",
            "algorithm_memory_kb",
            "algorithm_memory_min_kb",
            "algorithm_memory_max_kb",
            "peak_rss_kb",
            "peak_rss_min_kb",
            "peak_rss_max_kb",
            "maxrss_kb",
        ],
    )
    write_profiling_plot(rows)
    print({"profiling_cases": len(rows)})


if __name__ == "__main__":
    main()
