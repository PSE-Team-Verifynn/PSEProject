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

from qs_common import (
    QUALITY_OUT_DIR,
    ROOT,
    compute_polygon,
    convex_hull,
    ensure_output_dir,
    input_dim_from_model,
    maxrss_kb,
    polygon_area,
    run_model_samples,
    write_csv,
    write_json,
    write_quality_case_plot,
    write_quality_plot,
    write_quality_summary,
)

BOUNDS_SUITE_NAME = "bounds_variation"
NEURON_SUITE_NAME = "neuron_variation"
QUALITY_FIELDNAMES = [
    "case",
    "suite",
    "network_name",
    "bounds_label",
    "neurons_label",
    "plot_label",
    "runtime_ms",
    "memory_delta_kb",
    "containment_pass",
    "containment_ratio",
    "sample_point_containment_ratio",
    "avg_bound_width",
    "avg_sample_width",
    "avg_tightness_ratio",
    "polygon_area",
    "sample_hull_area",
    "polygon_over_hull_area_ratio",
    "min_slack",
    "avg_slack",
    "sample_metric_outputs",
    "bounded_directions",
    "checked_directions",
]


def suite_dir(name: str) -> Path:
    return QUALITY_OUT_DIR / name


def suite_cases_dir(name: str) -> Path:
    return suite_dir(name) / "cases"


def load_network_and_bounds(network_path, bounds_path=None, default_bounds=None):
    nn_res = NeuralNetworkLoader().load_neural_network(str(network_path))
    if not nn_res.is_success:
        raise nn_res.error

    network = nn_res.data
    input_dim = input_dim_from_model(network_path)
    if default_bounds is not None:
        bounds_map = {index: (float(default_bounds[0]), float(default_bounds[1])) for index in range(input_dim)}
    else:
        config = NetworkVerificationConfig(network, [input_dim])
        bounds_res = InputBoundsLoader().load_input_bounds(str(bounds_path), config)
        if not bounds_res.is_success:
            raise bounds_res.error
        bounds_map = bounds_res.data
    bounds_np = np.array([bounds_map[index] for index in range(len(bounds_map))], dtype=float)
    return network, bounds_map, bounds_np


def evaluate_selected_neuron_samples(network, selected_neurons: list[tuple[int, int]], samples: np.ndarray) -> np.ndarray:
    identity_directions = [(1.0, 0.0), (0.0, 1.0)]
    modified_model = NetworkModifier().custom_output_layer(network.model, selected_neurons, identity_directions)
    session = ort.InferenceSession(modified_model.SerializeToString(), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return run_model_samples(session, input_name, output_name, samples)


def make_case(
    suite: str,
    case: str,
    network_name: str,
    network_file: str,
    bounds_label: str,
    selected_neurons: list[tuple[int, int]],
    samples: int,
    *,
    bounds_file: str | None = None,
    default_bounds: tuple[float, float] | None = None,
) -> dict:
    neurons_label = f"{selected_neurons[0]} & {selected_neurons[1]}"
    control_label = bounds_label if suite == BOUNDS_SUITE_NAME else neurons_label
    return {
        "suite": suite,
        "case": case,
        "network_name": network_name,
        "network": ROOT / "TestFiles" / network_file,
        "bounds": None if bounds_file is None else ROOT / "TestFiles" / bounds_file,
        "default_bounds": default_bounds,
        "bounds_label": bounds_label,
        "selected_neurons": selected_neurons,
        "neurons_label": neurons_label,
        "plot_label": f"{network_name}\n{control_label}",
        "algorithm": ROOT / "algorithms" / "box_ibp_numpy.py",
        "samples": samples,
    }


def build_bounds_variation_cases() -> list[dict]:
    return [
        make_case(BOUNDS_SUITE_NAME, "NN1_B1_BoxIBP", "NN1", "NN1.onnx", "B1.csv", [(0, 0), (0, 1)], 2000, bounds_file="B1.csv"),
        make_case(BOUNDS_SUITE_NAME, "NN1_B2_BoxIBP", "NN1", "NN1.onnx", "B2.vnnlib", [(0, 0), (0, 1)], 2000, bounds_file="B2.vnnlib"),
        make_case(BOUNDS_SUITE_NAME, "NN1_UnitBox_BoxIBP", "NN1", "NN1.onnx", "[0,1]^n", [(0, 0), (0, 1)], 2000, default_bounds=(0.0, 1.0)),
        make_case(BOUNDS_SUITE_NAME, "IR11_1_B1_BoxIBP", "IR11_1", "IR11_1_gemm.onnx", "B1.csv", [(0, 0), (0, 1)], 1000, bounds_file="B1.csv"),
        make_case(BOUNDS_SUITE_NAME, "IR11_1_B2_BoxIBP", "IR11_1", "IR11_1_gemm.onnx", "B2.vnnlib", [(0, 0), (0, 1)], 1000, bounds_file="B2.vnnlib"),
        make_case(BOUNDS_SUITE_NAME, "IR11_1_UnitBox_BoxIBP", "IR11_1", "IR11_1_gemm.onnx", "[0,1]^n", [(0, 0), (0, 1)], 1000, default_bounds=(0.0, 1.0)),
        make_case(BOUNDS_SUITE_NAME, "IR11_2_UnitBox_BoxIBP", "IR11_2", "IR11_2_gemm.onnx", "[0,1]^n", [(1, 0), (1, 1)], 1000, default_bounds=(0.0, 1.0)),
        make_case(BOUNDS_SUITE_NAME, "IR11_2_Symmetric_BoxIBP", "IR11_2", "IR11_2_gemm.onnx", "[-1,1]^n", [(1, 0), (1, 1)], 1000, default_bounds=(-1.0, 1.0)),
        make_case(BOUNDS_SUITE_NAME, "IR11_2_Narrow_BoxIBP", "IR11_2", "IR11_2_gemm.onnx", "[0.25,0.75]^n", [(1, 0), (1, 1)], 1000, default_bounds=(0.25, 0.75)),
    ]


def build_neuron_variation_cases() -> list[dict]:
    return [
        make_case(NEURON_SUITE_NAME, "NN1_Neurons01_BoxIBP", "NN1", "NN1.onnx", "B2.vnnlib", [(0, 0), (0, 1)], 2000, bounds_file="B2.vnnlib"),
        make_case(NEURON_SUITE_NAME, "NN1_Neurons23_BoxIBP", "NN1", "NN1.onnx", "B2.vnnlib", [(0, 2), (0, 3)], 2000, bounds_file="B2.vnnlib"),
        make_case(NEURON_SUITE_NAME, "NN1_Output01_BoxIBP", "NN1", "NN1.onnx", "B2.vnnlib", [(1, 0), (1, 1)], 2000, bounds_file="B2.vnnlib"),
        make_case(NEURON_SUITE_NAME, "IR11_1_Neurons01_BoxIBP", "IR11_1", "IR11_1_gemm.onnx", "B2.vnnlib", [(0, 0), (0, 1)], 1000, bounds_file="B2.vnnlib"),
        make_case(NEURON_SUITE_NAME, "IR11_1_Neurons23_BoxIBP", "IR11_1", "IR11_1_gemm.onnx", "B2.vnnlib", [(0, 2), (0, 3)], 1000, bounds_file="B2.vnnlib"),
        make_case(NEURON_SUITE_NAME, "IR11_1_Output01_BoxIBP", "IR11_1", "IR11_1_gemm.onnx", "B2.vnnlib", [(1, 0), (1, 1)], 1000, bounds_file="B2.vnnlib"),
        make_case(NEURON_SUITE_NAME, "IR11_2_Hidden01_BoxIBP", "IR11_2", "IR11_2_gemm.onnx", "[0,1]^n", [(1, 0), (1, 1)], 1000, default_bounds=(0.0, 1.0)),
        make_case(NEURON_SUITE_NAME, "IR11_2_Hidden23_BoxIBP", "IR11_2", "IR11_2_gemm.onnx", "[0,1]^n", [(1, 2), (1, 3)], 1000, default_bounds=(0.0, 1.0)),
        make_case(NEURON_SUITE_NAME, "IR11_2_FirstLayer01_BoxIBP", "IR11_2", "IR11_2_gemm.onnx", "[0,1]^n", [(0, 0), (0, 1)], 1000, default_bounds=(0.0, 1.0)),
    ]


def evaluate_case(case: dict, case_output_dir: Path) -> dict:
    network, bounds_map, bounds_np = load_network_and_bounds(
        case["network"],
        case.get("bounds"),
        case.get("default_bounds"),
    )

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
    direction_session = ort.InferenceSession(modified_model.SerializeToString(), providers=["CPUExecutionProvider"])
    input_name = direction_session.get_inputs()[0].name
    output_name = direction_session.get_outputs()[0].name
    actual_outputs = run_model_samples(direction_session, input_name, output_name, samples)
    sample_points = evaluate_selected_neuron_samples(network, case["selected_neurons"], samples)

    actual_min = actual_outputs.min(axis=0)
    actual_max = actual_outputs.max(axis=0)
    computed_min = computed_bounds[:, 0]
    computed_max = computed_bounds[:, 1]
    contained = (actual_min >= computed_min - 1e-6) & (actual_max <= computed_max + 1e-6)
    width = computed_max - computed_min
    actual_width = actual_max - actual_min
    tightness = np.divide(actual_width, width, out=np.zeros_like(actual_width), where=width > 0)
    upper_slack = computed_max - actual_max
    lower_slack = actual_min - computed_min
    slack_values = np.concatenate([lower_slack, upper_slack])

    polygon = compute_polygon([(float(bound_low), float(bound_high)) for bound_low, bound_high in computed_bounds], directions)
    sample_hull = convex_hull(sample_points)
    polygon_area_value = polygon_area(polygon)
    sample_hull_area = polygon_area(sample_hull)
    polygon_area_ratio = polygon_area_value / sample_hull_area if sample_hull_area > 1e-12 else None

    projections = sample_points @ np.asarray(directions, dtype=np.float32).T
    point_inside = np.all(
        (projections >= computed_min - 1e-6) & (projections <= computed_max + 1e-6),
        axis=1,
    )
    point_containment_ratio = float(point_inside.mean()) if len(point_inside) else 0.0

    width_rows = []
    for direction_index, direction in enumerate(directions):
        width_rows.append(
            {
                "direction_index": direction_index,
                "direction": [round(float(direction[0]), 6), round(float(direction[1]), 6)],
                "computed_min": round(float(computed_min[direction_index]), 6),
                "computed_max": round(float(computed_max[direction_index]), 6),
                "actual_min": round(float(actual_min[direction_index]), 6),
                "actual_max": round(float(actual_max[direction_index]), 6),
                "computed_width": round(float(width[direction_index]), 6),
                "sample_width": round(float(actual_width[direction_index]), 6),
                "lower_slack": round(float(lower_slack[direction_index]), 6),
                "upper_slack": round(float(upper_slack[direction_index]), 6),
                "contained": bool(contained[direction_index]),
            }
        )

    sample_summary = run_samples_for_bounds(network, list(bounds_map.values()), min(case["samples"], 1000), ["max", "mean", "range"])

    detail_payload = {
        "case": case["case"],
        "suite": case["suite"],
        "network_name": case["network_name"],
        "bounds_label": case["bounds_label"],
        "neurons_label": case["neurons_label"],
        "plot_label": case["plot_label"],
        "network": case["network"].name,
        "bounds": case["bounds"].name if case.get("bounds") else None,
        "default_bounds": list(case["default_bounds"]) if case.get("default_bounds") else None,
        "algorithm": case["algorithm"].name,
        "selected_neurons": case["selected_neurons"],
        "samples": int(case["samples"]),
        "polygon_vertices": [[round(float(x), 6), round(float(y), 6)] for x, y in polygon],
        "sample_hull_vertices": [[round(float(x), 6), round(float(y), 6)] for x, y in sample_hull],
        "sample_point_preview": [[round(float(point[0]), 6), round(float(point[1]), 6)] for point in sample_points[:250]],
        "direction_metrics": width_rows,
    }
    write_json(case_output_dir / f"{case['case']}.json", detail_payload)
    write_quality_case_plot(case["case"], sample_points, polygon, sample_hull, width_rows, case_output_dir / f"{case['case']}.png")

    return {
        "case": case["case"],
        "suite": case["suite"],
        "network_name": case["network_name"],
        "bounds_label": case["bounds_label"],
        "neurons_label": case["neurons_label"],
        "plot_label": case["plot_label"],
        "runtime_ms": round(runtime_ms, 3),
        "memory_delta_kb": int(memory_delta_kb),
        "containment_pass": bool(contained.all()),
        "containment_ratio": round(float(contained.mean()), 4),
        "avg_bound_width": round(float(width.mean()), 6),
        "avg_sample_width": round(float(actual_width.mean()), 6),
        "avg_tightness_ratio": round(float(tightness.mean()), 6),
        "sample_point_containment_ratio": round(point_containment_ratio, 4),
        "polygon_area": round(float(polygon_area_value), 6),
        "sample_hull_area": round(float(sample_hull_area), 6),
        "polygon_over_hull_area_ratio": None if polygon_area_ratio is None else round(float(polygon_area_ratio), 6),
        "min_slack": round(float(slack_values.min()), 6),
        "avg_slack": round(float(slack_values.mean()), 6),
        "sample_metric_outputs": len(sample_summary["outputs"]),
        "bounded_directions": int(contained.sum()),
        "checked_directions": int(len(contained)),
    }


def run_quality_suite(suite_name: str, cases: list[dict]) -> list[dict]:
    suite_output_dir = suite_dir(suite_name)
    suite_case_output_dir = suite_cases_dir(suite_name)
    ensure_output_dir(suite_output_dir)
    ensure_output_dir(suite_case_output_dir)

    rows = [evaluate_case(case, suite_case_output_dir) for case in cases]
    write_json(suite_output_dir / "quality_results.json", rows)
    write_csv(suite_output_dir / "quality_metrics.csv", rows, QUALITY_FIELDNAMES)
    write_quality_plot(rows, suite_output_dir / "quality_metrics.png")
    write_quality_summary(rows, suite_output_dir / "quality_info.txt", suite_name)
    return rows


def main() -> None:
    Storage().num_directions = 16
    np.random.seed(7)
    random.seed(7)

    ensure_output_dir(QUALITY_OUT_DIR)
    bounds_rows = run_quality_suite(BOUNDS_SUITE_NAME, build_bounds_variation_cases())
    neuron_rows = run_quality_suite(NEURON_SUITE_NAME, build_neuron_variation_cases())

    summary = {
        "bounds_variation_cases": len(bounds_rows),
        "neuron_variation_cases": len(neuron_rows),
    }
    write_json(QUALITY_OUT_DIR / "quality_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
