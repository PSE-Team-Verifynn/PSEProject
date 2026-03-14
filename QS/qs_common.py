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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

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


def write_quality_plot(rows: list[dict], output_path: Path) -> None:
    if rows and all(row.get("suite") == "neuron_variation" for row in rows):
        write_neuron_variation_plot(rows, output_path)
        return

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
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_neuron_variation_plot(rows: list[dict], output_path: Path) -> None:
    metrics = [
        ("containment_ratio", "Directional containment ratio", "share of bounded directions"),
        ("sample_point_containment_ratio", "Sample point containment ratio", "share of sampled points inside polygon"),
        ("avg_tightness_ratio", "Average tightness ratio", "sample width / computed width"),
        ("polygon_over_hull_area_ratio", "Polygon / hull area ratio", "overapproximation factor"),
    ]
    networks = list(dict.fromkeys(row["network_name"] for row in rows))
    group_order = ["same_layer", "cross_layer"]
    group_labels = {"same_layer": "same layer", "cross_layer": "cross layer"}
    group_colors = {"same_layer": "#2563eb", "cross_layer": "#f97316"}

    aggregated: dict[tuple[str, str], dict[str, float]] = {}
    for network in networks:
        network_rows = [row for row in rows if row["network_name"] == network]
        for group in group_order:
            selected_rows = [row for row in network_rows if row.get("pair_group") == group]
            if not selected_rows:
                continue
            aggregated[(network, group)] = {}
            for metric_key, _, _ in metrics:
                values = [row[metric_key] for row in selected_rows if row[metric_key] is not None]
                aggregated[(network, group)][metric_key] = float(np.mean(values)) if values else 0.0

    x = np.arange(len(networks), dtype=float)
    width = 0.28
    offsets = {"same_layer": -width / 2, "cross_layer": width / 2}
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8))

    for axis, (metric_key, title, ylabel) in zip(axes.flat, metrics):
        for group in group_order:
            heights = [aggregated.get((network, group), {}).get(metric_key, 0.0) for network in networks]
            axis.bar(
                x + offsets[group],
                heights,
                width=width,
                color=group_colors[group],
                label=group_labels[group],
            )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x)
        axis.set_xticklabels(networks)
        if "containment" in metric_key:
            axis.set_ylim(0, 1.1)
        axis.grid(axis="y", alpha=0.15)

    axes[0, 0].legend(loc="best")
    fig.suptitle("Neuron variation grouped by pair type", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
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


def write_quality_case_plot(
    case_name: str,
    sample_points: np.ndarray,
    polygon: list[tuple[float, float]],
    sample_hull: list[tuple[float, float]],
    width_rows: list[dict],
    output_path: Path,
) -> None:
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
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def write_quality_summary(rows: list[dict], output_path: Path, suite_name: str) -> None:
    lines = [
        "Quality QS information",
        "",
        f"Suite: {suite_name}",
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
        "- quality_results.json",
        "- quality_metrics.csv",
        "- quality_metrics.png",
        "- cases/<case>.png",
        "- cases/<case>.json",
        "",
        "Cases:",
    ]
    for row in rows:
        bounds_label = row.get("bounds_label")
        if bounds_label:
            lines.append(f"- {row['case']} ({bounds_label})")
        else:
            lines.append(f"- {row['case']}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_profiling_plot(rows: list[dict]) -> None:
    labels = [row["case"] for row in rows]
    load_runtime_ms = [row["load_runtime_ms"] for row in rows]
    algorithm_runtime_ms = [row["algorithm_runtime_ms"] for row in rows]
    load_memory_mb = [round(row["load_memory_kb"] / 1024, 3) for row in rows]
    algorithm_memory_mb = [round(row["algorithm_memory_kb"] / 1024, 3) for row in rows]
    baseline_memory_mb = [round(row["baseline_rss_kb"] / 1024, 3) for row in rows]
    peak_memory_mb = [round(row["peak_rss_kb"] / 1024, 3) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    axes[0].bar(labels, load_runtime_ms, color="#60a5fa", label="Load")
    axes[0].bar(labels, algorithm_runtime_ms, bottom=load_runtime_ms, color="#2563eb", label="Algorithm")
    axes[0].set_title("Runtime split by network")
    axes[0].set_ylabel("ms")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].legend(loc="best")

    axes[1].bar(labels, baseline_memory_mb, color="#cbd5e1", label="Program baseline")
    axes[1].bar(labels, load_memory_mb, bottom=baseline_memory_mb, color="#f59e0b", label="Model load delta")
    stacked_memory_mb = [baseline + load for baseline, load in zip(baseline_memory_mb, load_memory_mb)]
    axes[1].bar(labels, algorithm_memory_mb, bottom=stacked_memory_mb, color="#dc2626", label="Algorithm delta")
    axes[1].plot(labels, peak_memory_mb, marker="o", color="#111827", linewidth=2, label="Peak RSS")
    axes[1].set_title("Memory split by network")
    axes[1].set_ylabel("MB")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend(loc="best")

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


def write_gui_status_diagram(rows: list[dict], output_path: Path) -> None:
    count = max(len(rows), 1)
    top_margin = 2.4
    bottom_margin = 0.8
    row_gap = 1.08
    first_row_y = bottom_margin + row_gap * (count - 1)
    fig_height = max(7.8, first_row_y + top_margin + 1.2)
    fig, axis = plt.subplots(figsize=(12.4, fig_height))
    background = "#f1f5f9"
    axis_width = 12.0
    axis_height = first_row_y + top_margin + 0.9

    fig.patch.set_facecolor(background)
    axis.set_facecolor(background)
    axis.set_xlim(0, axis_width)
    axis.set_ylim(0, axis_height)
    axis.axis("off")

    # Soft page panel to give the diagram structure without crowding the first row.
    axis.add_patch(
        FancyBboxPatch(
            (0.45, 0.28),
            axis_width - 0.9,
            axis_height - 0.52,
            boxstyle="round,pad=0.18,rounding_size=0.22",
            linewidth=1.2,
            edgecolor="#cbd5e1",
            facecolor="#ffffff",
            zorder=0,
        )
    )

    axis.text(
        0.95,
        axis_height - 0.8,
        "GUI Smoke Test Summary",
        fontsize=21,
        fontweight="bold",
        color="#0f172a",
        ha="left",
        va="center",
    )
    axis.text(
        0.95,
        axis_height - 1.32,
        "Each executed check is shown with its final status.",
        fontsize=10.5,
        color="#475569",
        ha="left",
        va="center",
    )

    x_start = 1.1
    box_width = 9.8
    box_height = 0.74

    for index, row in enumerate(rows):
        y = first_row_y - index * row_gap
        passed = bool(row["passed"])
        border = "#16a34a" if passed else "#dc2626"
        fill = "#ecfdf3" if passed else "#fef2f2"
        accent = "#166534" if passed else "#991b1b"
        status = "passed" if passed else "failed"
        label = row["check"].replace("_", " ").title()

        if index < len(rows) - 1:
            next_y = first_row_y - (index + 1) * row_gap
            axis.add_patch(
                FancyArrowPatch(
                    (x_start + box_width / 2, y - box_height / 2 - 0.07),
                    (x_start + box_width / 2, next_y + box_height / 2 + 0.07),
                    arrowstyle="-|>",
                    mutation_scale=13,
                    linewidth=1.5,
                    color="#cbd5e1",
                    zorder=1,
                )
            )

        axis.add_patch(
            FancyBboxPatch(
                (x_start, y - box_height / 2),
                box_width,
                box_height,
                boxstyle="round,pad=0.15,rounding_size=0.13",
                linewidth=1.6,
                edgecolor=border,
                facecolor=fill,
                zorder=2,
            )
        )

        axis.add_patch(
            FancyBboxPatch(
                (x_start + box_width - 1.62, y - 0.23),
                1.18,
                0.46,
                boxstyle="round,pad=0.08,rounding_size=0.18",
                linewidth=0,
                edgecolor=accent,
                facecolor="#dcfce7" if passed else "#fee2e2",
                zorder=3,
            )
        )

        axis.text(
            x_start + 0.36,
            y,
            label,
            fontsize=12.2,
            color="#0f172a",
            va="center",
            ha="left",
            zorder=4,
        )
        axis.text(
            x_start + box_width - 1.03,
            y,
            status,
            fontsize=10.6,
            fontweight="bold",
            color=accent,
            va="center",
            ha="center",
            zorder=4,
        )

    fig.subplots_adjust(left=0.035, right=0.985, top=0.99, bottom=0.03)
    fig.savefig(output_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_gui_outputs(rows: list[dict]) -> None:
    write_json(GUI_OUT_DIR / "gui_smoke_results.json", rows)
    write_gui_testplan(rows)
    write_gui_status_diagram(rows, GUI_OUT_DIR / "gui_status_diagram.png")


def write_gui_performance_plot(payload: dict, output_path: Path) -> None:
    startup_ms = float(payload["startup_time_ms"])
    startup_memory_mb = round(float(payload.get("startup_memory_kb", 0)) / 1024, 3)
    load_rows = payload.get("network_load_results", [])
    labels = ["gui_startup"] + [row["case"] for row in load_rows]
    time_values_ms = [startup_ms] + [float(row["load_to_display_ms"]) for row in load_rows]
    memory_values_mb = [startup_memory_mb] + [round(float(row.get("load_memory_kb", 0)) / 1024, 3) for row in load_rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    time_bars = axes[0].bar(labels, time_values_ms, color="#2563eb", width=0.62)
    axes[0].set_title("Startup vs Network Load-to-Display")
    axes[0].set_ylabel("ms")
    axes[0].grid(axis="y", alpha=0.15)
    axes[0].tick_params(axis="x", rotation=20)
    for bar, value in zip(time_bars, time_values_ms):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}", ha="center", va="bottom", fontsize=9.5)

    memory_bars = axes[1].bar(labels, memory_values_mb, color="#dc2626", width=0.62)
    axes[1].set_title("Startup vs Network Load Memory Delta")
    axes[1].set_ylabel("MB")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.15)
    for bar, value in zip(memory_bars, memory_values_mb):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}", ha="center", va="bottom", fontsize=9.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_gui_performance_outputs(payload: dict) -> None:
    write_json(GUI_OUT_DIR / "gui_performance_results.json", payload)
    write_gui_performance_plot(payload, GUI_OUT_DIR / "gui_performance_diagram.png")
    lines = [
        "GUI performance runtime information",
        "",
        f"Mode: {payload['mode']}",
        f"Startup target: {payload['startup_target']}",
        f"Network load target: {payload.get('network_load_target', 'network_loaded_and_displayed')}",
        f"Startup time (ms): {payload['startup_time_ms']}",
        f"Startup memory delta (kB): {payload.get('startup_memory_kb', 0)}",
        "",
        "What this measures for startup:",
        "- QApplication creation",
        "- style and color manager setup",
        "- MainWindow construction",
        "- initial show call and first processed GUI events",
        "",
        "What this measures for network load:",
        "- one fresh GUI process per network case",
        "- ONNX load and validation",
        "- tab creation and network scene construction",
        "- GUI event processing until the network can be displayed",
        "",
        "Per-network load measurements:",
    ]
    for row in payload["network_load_results"]:
        lines.append(
            f"- {row['case']}: {row['load_to_display_ms']} ms, load memory delta={row.get('load_memory_kb', 0)} kB, "
            f"scene items={row['scene_items']}"
        )
    lines.extend([
        "",
        "Result files:",
        "- QS/GUI/gui_performance_results.json",
        "- QS/GUI/gui_performance_diagram.png",
    ])
    (GUI_OUT_DIR / "gui_performance_info.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
