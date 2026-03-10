from __future__ import annotations

import csv
import json
import os
import resource
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
QS_DIR = ROOT / "QS"
GUI_OUT_DIR = QS_DIR / "GUI"
QUALITY_OUT_DIR = QS_DIR / "QUALITY"
PROFILING_OUT_DIR = QS_DIR / "PROFILING"


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maxrss_kb() -> int:
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.uname().sysname == "Darwin":
        return int(maxrss / 1024)
    return int(maxrss)


def input_dim_from_model(model_path: Path) -> int:
    model = onnx.load(str(model_path))
    dim = model.graph.input[0].type.tensor_type.shape.dim[-1].dim_value
    if not dim:
        raise ValueError(f"Could not determine input dimension for {model_path}")
    return int(dim)


def run_model_samples(session: ort.InferenceSession, input_name: str, output_name: str, samples: np.ndarray) -> np.ndarray:
    input_shape = session.get_inputs()[0].shape
    first_dim = input_shape[0] if input_shape else None
    if first_dim == 1:
        outputs = [session.run([output_name], {input_name: samples[index:index + 1]})[0] for index in range(samples.shape[0])]
        return np.concatenate(outputs, axis=0)
    return session.run([output_name], {input_name: samples})[0]


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
    fig.savefig(QUALITY_OUT_DIR / "quality_metrics.png", dpi=160)
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
    fig.savefig(PROFILING_OUT_DIR / "profiling_metrics.png", dpi=160)
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
    lines.append("- TS1 in QS/TS1/ts1.txt")
    lines.append("- TS2 in QS/TS2/ts2.txt")
    (GUI_OUT_DIR / "gui_testplan.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
