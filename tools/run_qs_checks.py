from __future__ import annotations

import csv
import json
import os
import random
import resource
import time
from multiprocessing import get_context
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxruntime as ort
from PySide6.QtWidgets import QApplication, QWidget
from nn_verification_visualisation import resources_rc

from nn_verification_visualisation.controller.process_manager.algorithm_executor import AlgorithmExecutor
from nn_verification_visualisation.controller.process_manager.network_modifier import NetworkModifier
from nn_verification_visualisation.controller.process_manager.sample_runner import run_samples_for_bounds
from nn_verification_visualisation.model.data.algorithm import Algorithm
from nn_verification_visualisation.model.data.diagram_config import DiagramConfig
from nn_verification_visualisation.model.data.network_verification_config import NetworkVerificationConfig
from nn_verification_visualisation.model.data.plot_generation_config import PlotGenerationConfig
from nn_verification_visualisation.model.data.storage import Storage
from nn_verification_visualisation.model.data_loader.input_bounds_loader import InputBoundsLoader
from nn_verification_visualisation.model.data_loader.neural_network_loader import NeuralNetworkLoader
from nn_verification_visualisation.view.base_view.color_manager import ColorManager
from nn_verification_visualisation.view.base_view.main_window import MainWindow


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "TestResults" / "QS"


def _maxrss_kb() -> int:
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.uname().sysname == "Darwin":
        return int(maxrss / 1024)
    return int(maxrss)


def _input_dim_from_model(model_path: Path) -> int:
    model = onnx.load(str(model_path))
    dim = model.graph.input[0].type.tensor_type.shape.dim[-1].dim_value
    if not dim:
        raise ValueError(f"Could not determine input dimension for {model_path}")
    return int(dim)


def _load_network_and_bounds(network_path: Path, bounds_path: Path | None) -> tuple:
    nn_res = NeuralNetworkLoader().load_neural_network(str(network_path))
    if not nn_res.is_success:
        raise nn_res.error

    network = nn_res.data
    input_dim = _input_dim_from_model(network_path)
    config = NetworkVerificationConfig(network, [input_dim])

    if bounds_path is None:
        bounds_map = {index: (0.0, 1.0) for index in range(input_dim)}
    else:
        bounds_res = InputBoundsLoader().load_input_bounds(str(bounds_path), config)
        if not bounds_res.is_success:
            raise bounds_res.error
        bounds_map = bounds_res.data

    bounds_np = np.array([bounds_map[index] for index in range(len(bounds_map))], dtype=float)
    return network, bounds_map, bounds_np


def _run_model_samples(session: ort.InferenceSession, input_name: str, output_name: str, samples: np.ndarray) -> np.ndarray:
    input_shape = session.get_inputs()[0].shape
    first_dim = input_shape[0] if input_shape else None
    if first_dim == 1:
        outputs = [session.run([output_name], {input_name: samples[index:index + 1]})[0] for index in range(samples.shape[0])]
        return np.concatenate(outputs, axis=0)
    return session.run([output_name], {input_name: samples})[0]


def run_gui_smoke_tests() -> list[dict]:
    app = QApplication.instance() or QApplication([])
    color_manager = ColorManager(app)
    color_manager.load_raw(":src/nn_verification_visualisation/style.qss")

    with patch("nn_verification_visualisation.controller.input_manager.plot_view_controller.AlgorithmFileObserver", lambda *args, **kwargs: None):
        window = MainWindow(color_manager)
        color_manager.main_window = window
        color_manager.set_colors(ColorManager.NETWORK_COLORS)

    results: list[dict] = []
    results.append(
        {
            "check": "window_init",
            "passed": window.windowTitle() == "PSE Neuron App" and window.base_view.stack.currentIndex() == 0,
            "details": {
                "title": window.windowTitle(),
                "stack_index": window.base_view.stack.currentIndex(),
            },
        }
    )

    window.base_view.change_active_view()
    plot_ok = window.base_view.active_view is window.base_view.plot_view and window.base_view.stack.currentIndex() == 1
    window.base_view.change_active_view()
    back_ok = window.base_view.active_view is window.base_view.network_view and window.base_view.stack.currentIndex() == 0
    results.append(
        {
            "check": "view_switching",
            "passed": plot_ok and back_ok,
            "details": {"plot_ok": plot_ok, "back_ok": back_ok},
        }
    )

    class DummyPlotWidget(QWidget):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def render_plot(self, *args, **kwargs):
            return None

    class DummyPlotSettingsWidget(QWidget):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def set_selection(self, _selection):
            return None

    class DummyNNConfig:
        pass

    nn_config = DummyNNConfig()
    algorithm = Algorithm(name="A", path="algorithms/box_ibp_numpy.py", is_deterministic=True)
    config_fail = PlotGenerationConfig(nnconfig=nn_config, algorithm=algorithm, selected_neurons=[(0, 0)], parameters=[], bounds_index=-1)
    config_ok = PlotGenerationConfig(nnconfig=nn_config, algorithm=algorithm, selected_neurons=[(0, 1)], parameters=[], bounds_index=-1)
    diagram = DiagramConfig(
        plot_generation_configs=[config_fail, config_ok],
        polygons=[None, [(0.0, 0.0), (1.0, 0.0)]],
    )
    diagram.plots = [[0]]

    loading_widget = QWidget()
    loading_widget.diagram_config = diagram
    window.base_view.plot_view.tabs.addTab(loading_widget, "Loading...")
    Storage().diagrams = []

    with patch("nn_verification_visualisation.view.plot_view.plot_page.PlotWidget", DummyPlotWidget), patch(
        "nn_verification_visualisation.view.plot_view.plot_page.PlotSettingsWidget", DummyPlotSettingsWidget
    ), patch.object(Storage, "request_autosave", lambda self: None):
        controller = window.base_view.plot_view.controller
        controller.create_diagram_tab(loading_widget)

    replaced_widget = window.base_view.plot_view.tabs.widget(0)
    results.append(
        {
            "check": "diagram_tab_creation",
            "passed": len(Storage().diagrams) == 1
            and len(Storage().diagrams[0].polygons) == 1
            and replaced_widget.__class__.__name__ == "PlotPage",
            "details": {
                "stored_diagrams": len(Storage().diagrams),
                "remaining_polygons": len(Storage().diagrams[0].polygons),
                "tab_widget": replaced_widget.__class__.__name__,
            },
        }
    )

    window.close()
    app.quit()
    return results


def run_quality_checks() -> list[dict]:
    Storage().num_directions = 16
    np.random.seed(7)
    random.seed(7)

    cases = [
        {
            "case": "NN1_BoxIBP",
            "network": ROOT / "TestFiles" / "NN1.onnx",
            "bounds": ROOT / "TestFiles" / "B1.csv",
            "algorithm": ROOT / "algorithms" / "box_ibp_numpy.py",
            "selected_neurons": [(0, 0), (0, 1)],
            "samples": 2000,
        },
        {
            "case": "NN1_Zonotope",
            "network": ROOT / "TestFiles" / "NN1.onnx",
            "bounds": ROOT / "TestFiles" / "B1.csv",
            "algorithm": ROOT / "algorithms" / "simple_zonotope.py",
            "selected_neurons": [(0, 0), (0, 1)],
            "samples": 2000,
        },
        {
            "case": "NN2_BoxIBP",
            "network": ROOT / "TestFiles" / "NN2.onnx",
            "bounds": ROOT / "TestFiles" / "B1.csv",
            "algorithm": ROOT / "algorithms" / "box_ibp_numpy.py",
            "selected_neurons": [(0, 99), (0, 199)],
            "samples": 1000,
        },
    ]

    results: list[dict] = []
    for case in cases:
        network, bounds_map, bounds_np = _load_network_and_bounds(case["network"], case["bounds"])

        start_rss = _maxrss_kb()
        start_time = time.perf_counter()
        result = AlgorithmExecutor().execute_algorithm(network.model, bounds_np, str(case["algorithm"]), case["selected_neurons"])
        runtime_ms = (time.perf_counter() - start_time) * 1000
        memory_delta_kb = max(0, _maxrss_kb() - start_rss)
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
        actual_outputs = _run_model_samples(session, input_name, output_name, samples)

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
            }
        )

    return results


def _profile_case_worker(network_path: str, algorithm_path: str, selected_neurons: list[tuple[int, int]], queue) -> None:
    try:
        Storage().num_directions = 16
        network, _, bounds_np = _load_network_and_bounds(Path(network_path), None)
        start_time = time.perf_counter()
        result = AlgorithmExecutor().execute_algorithm(network.model, bounds_np, algorithm_path, selected_neurons)
        runtime_ms = (time.perf_counter() - start_time) * 1000
        if not result.is_success:
            raise result.error
        queue.put({"runtime_ms": round(runtime_ms, 3), "maxrss_kb": _maxrss_kb()})
    except BaseException as error:
        queue.put({"error": repr(error)})


def run_profiling_checks() -> list[dict]:
    cases = [
        {
            "case": "NN1_small",
            "network": ROOT / "TestFiles" / "NN1.onnx",
            "algorithm": ROOT / "algorithms" / "box_ibp_numpy.py",
            "selected_neurons": [(0, 0), (0, 1)],
        },
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
    ]

    ctx = get_context("spawn")
    results: list[dict] = []
    for case in cases:
        queue = ctx.Queue()
        process = ctx.Process(
            target=_profile_case_worker,
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


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_quality_plot(rows: list[dict]) -> None:
    labels = [row["case"] for row in rows]
    tightness = [row["avg_tightness_ratio"] for row in rows]
    containment = [row["containment_ratio"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(labels, containment, color=["#3b82f6", "#10b981", "#f59e0b"])
    axes[0].set_ylim(0, 1.1)
    axes[0].set_title("Containment ratio")
    axes[0].set_ylabel("share of bounded directions")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(labels, tightness, color=["#ef4444", "#8b5cf6", "#14b8a6"])
    axes[1].set_title("Average tightness ratio")
    axes[1].set_ylabel("sample width / computed width")
    axes[1].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "quality_metrics.png", dpi=160)
    plt.close(fig)


def write_profiling_plot(rows: list[dict]) -> None:
    labels = [row["case"] for row in rows]
    runtime_ms = [row["runtime_ms"] for row in rows]
    memory_mb = [round(row["maxrss_kb"] / 1024, 3) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(labels, runtime_ms, marker="o", color="#2563eb")
    axes[0].set_title("Runtime by network")
    axes[0].set_ylabel("ms")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].plot(labels, memory_mb, marker="o", color="#dc2626")
    axes[1].set_title("Peak memory by network")
    axes[1].set_ylabel("MB")
    axes[1].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "profiling_metrics.png", dpi=160)
    plt.close(fig)


def write_gui_testplan(rows: list[dict]) -> None:
    lines = [
        "GUI test plan execution",
        "",
        "Executed checks:",
    ]
    for row in rows:
        status = "passed" if row["passed"] else "failed"
        lines.append(f"- {row['check']}: {status}")
    lines.append("")
    lines.append("Manual scenario references:")
    lines.append("- TS1 in TestResults/TS1/ts1.txt")
    lines.append("- TS2 in TestResults/TS2/ts2.txt")
    (OUT_DIR / "gui_testplan.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gui_rows = run_gui_smoke_tests()
    quality_rows = run_quality_checks()
    profiling_rows = run_profiling_checks()

    (OUT_DIR / "gui_smoke_results.json").write_text(json.dumps(gui_rows, indent=2), encoding="utf-8")
    (OUT_DIR / "quality_results.json").write_text(json.dumps(quality_rows, indent=2), encoding="utf-8")
    (OUT_DIR / "profiling_results.json").write_text(json.dumps(profiling_rows, indent=2), encoding="utf-8")

    write_csv(
        OUT_DIR / "quality_metrics.csv",
        quality_rows,
        ["case", "runtime_ms", "memory_delta_kb", "containment_pass", "containment_ratio", "avg_bound_width", "avg_sample_width", "avg_tightness_ratio", "sample_metric_outputs"],
    )
    write_csv(
        OUT_DIR / "profiling_metrics.csv",
        profiling_rows,
        ["case", "runtime_ms", "maxrss_kb"],
    )

    write_quality_plot(quality_rows)
    write_profiling_plot(profiling_rows)
    write_gui_testplan(gui_rows)

    summary = {
        "gui_checks_passed": all(row["passed"] for row in gui_rows),
        "quality_cases": len(quality_rows),
        "profiling_cases": len(profiling_rows),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
