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
QUALITY_CASES_OUT_DIR = QUALITY_OUT_DIR / "cases"
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
    labels = [row.get("plot_label", row["case"]) for row in rows]
    tightness = [row["avg_tightness_ratio"] for row in rows]
    containment = [row["containment_ratio"] for row in rows]
    point_containment = [row["sample_point_containment_ratio"] for row in rows]
    area_ratio = [0.0 if row["polygon_over_hull_area_ratio"] is None else row["polygon_over_hull_area_ratio"] for row in rows]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    colors = ["#3b82f6", "#10b981", "#f59e0b", "#f97316", "#06b6d4", "#84cc16", "#8b5cf6", "#ef4444", "#f43f5e", "#14b8a6"]
    bar_colors = [colors[index % len(colors)] for index in range(len(labels))]

    axes[0, 0].bar(labels, containment, color=bar_colors)
    axes[0, 0].set_ylim(0, 1.1)
    axes[0, 0].set_title("Directional containment ratio")
    axes[0, 0].set_ylabel("share of bounded directions")
    axes[0, 0].tick_params(axis="x", rotation=20)

    axes[0, 1].bar(labels, point_containment, color=bar_colors)
    axes[0, 1].set_ylim(0, 1.1)
    axes[0, 1].set_title("Sample point containment ratio")
    axes[0, 1].set_ylabel("share of sampled points inside polygon")
    axes[0, 1].tick_params(axis="x", rotation=20)

    axes[1, 0].bar(labels, tightness, color=bar_colors)
    axes[1, 0].set_title("Average tightness ratio")
    axes[1, 0].set_ylabel("sample width / computed width")
    axes[1, 0].tick_params(axis="x", rotation=20)

    axes[1, 1].bar(labels, area_ratio, color=bar_colors)
    axes[1, 1].set_title("Polygon / hull area ratio")
    axes[1, 1].set_ylabel("overapproximation factor")
    axes[1, 1].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(QUALITY_OUT_DIR / "quality_metrics.png", dpi=160)
    plt.close(fig)


def compute_polygon(bounds: list[tuple[float, float]], directions: list[tuple[float, float]]) -> list[tuple[float, float]]:
    def clip_polygon(poly: list[tuple[float, float]], a: float, b: float, c: float):
        def inside(point: tuple[float, float]) -> bool:
            return a * point[0] + b * point[1] <= c + 1e-9

        def intersect(point1: tuple[float, float], point2: tuple[float, float]):
            x1, y1 = point1
            x2, y2 = point2
            dx = x2 - x1
            dy = y2 - y1
            denom = a * dx + b * dy
            if abs(denom) < 1e-12:
                return point2
            t = (c - a * x1 - b * y1) / denom
            return (x1 + t * dx, y1 + t * dy)

        out: list[tuple[float, float]] = []
        for index in range(len(poly)):
            curr = poly[index]
            prev = poly[index - 1]
            curr_in = inside(curr)
            prev_in = inside(prev)
            if curr_in:
                if not prev_in:
                    out.append(intersect(prev, curr))
                out.append(curr)
            elif prev_in:
                out.append(intersect(prev, curr))
        return out

    max_bound = max(abs(value) for (low, high) in bounds for value in (low, high))
    margin = max(5.0, max_bound * 2.0 + 1.0)
    poly: list[tuple[float, float]] = [(-margin, -margin), (margin, -margin), (margin, margin), (-margin, margin)]

    for index, (low, high) in enumerate(bounds):
        a, b = directions[index]
        poly = clip_polygon(poly, a, b, high)
        if not poly:
            break
        poly = clip_polygon(poly, -a, -b, -low)
        if not poly:
            break
    return poly


def polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        area += point[0] * next_point[1] - next_point[0] * point[1]
    return abs(area) * 0.5


def convex_hull(points: np.ndarray) -> list[tuple[float, float]]:
    unique_points = sorted({(float(point[0]), float(point[1])) for point in points})
    if len(unique_points) <= 1:
        return unique_points

    def cross(origin, point_a, point_b) -> float:
        return (point_a[0] - origin[0]) * (point_b[1] - origin[1]) - (point_a[1] - origin[1]) * (point_b[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(unique_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def write_quality_case_plot(case_name: str, sample_points: np.ndarray, polygon: list[tuple[float, float]], sample_hull: list[tuple[float, float]], width_rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    axes[0].scatter(sample_points[:, 0], sample_points[:, 1], s=8, alpha=0.35, color="#2563eb", label="Samples")
    if polygon:
        polygon_closed = polygon + [polygon[0]]
        polygon_x = [point[0] for point in polygon_closed]
        polygon_y = [point[1] for point in polygon_closed]
        axes[0].fill(polygon_x, polygon_y, color="#f59e0b", alpha=0.18, label="Overapprox polygon")
        axes[0].plot(polygon_x, polygon_y, color="#ea580c", linewidth=2)
    if sample_hull:
        hull_closed = sample_hull + [sample_hull[0]]
        hull_x = [point[0] for point in hull_closed]
        hull_y = [point[1] for point in hull_closed]
        axes[0].plot(hull_x, hull_y, color="#059669", linewidth=2, label="Sample hull")
    axes[0].set_title(f"{case_name}: polygon vs samples")
    axes[0].set_xlabel("selected neuron 1")
    axes[0].set_ylabel("selected neuron 2")
    axes[0].legend(loc="best")

    direction_ids = [row["direction_index"] for row in width_rows]
    computed = [row["computed_width"] for row in width_rows]
    actual = [row["sample_width"] for row in width_rows]
    axes[1].plot(direction_ids, computed, marker="o", color="#dc2626", label="Computed width")
    axes[1].plot(direction_ids, actual, marker="o", color="#16a34a", label="Sample width")
    axes[1].set_title(f"{case_name}: width by direction")
    axes[1].set_xlabel("direction index")
    axes[1].set_ylabel("width")
    axes[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(QUALITY_CASES_OUT_DIR / f"{case_name}.png", dpi=170)
    plt.close(fig)


def write_quality_summary(rows: list[dict]) -> None:
    lines = [
        "Quality QS information",
        "",
        "Measured quality values:",
        "- containment_ratio: share of directions whose sampled extrema stay inside the computed bounds",
        "- sample_point_containment_ratio: share of sampled 2D neuron points that stay inside the computed polygon",
        "- avg_tightness_ratio: sample width divided by computed bound width",
        "- polygon_area: area of the halfspace polygon built from the computed bounds",
        "- sample_hull_area: convex hull area of the sampled 2D neuron outputs",
        "- polygon_over_hull_area_ratio: geometric overapproximation factor",
        "- min_slack: worst signed margin between computed and sampled directional bounds",
        "- avg_slack: mean signed directional slack across lower and upper bounds",
        "",
        "Generated artifacts:",
        "- QS/QUALITY/quality_results.json",
        "- QS/QUALITY/quality_metrics.csv",
        "- QS/QUALITY/quality_metrics.png",
        "- QS/QUALITY/cases/<case>.png",
        "- QS/QUALITY/cases/<case>.json",
        "",
        "Cases:",
    ]
    for row in rows:
        bounds_label = row.get("bounds_label")
        if bounds_label:
            lines.append(f"- {row['case']} ({bounds_label})")
        else:
            lines.append(f"- {row['case']}")
    (QUALITY_OUT_DIR / "quality_info.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
