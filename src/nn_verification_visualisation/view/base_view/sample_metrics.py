from __future__ import annotations

from typing import Any, Iterable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QWidget,
    QLabel,
    QVBoxLayout,
    QGridLayout,
    QScrollArea,
    QHBoxLayout,
    QComboBox,
)

from nn_verification_visualisation.controller.process_manager.sample_metric_registry import get_metric_map


class SampleMetricsWidget(QGroupBox):
    def __init__(
        self,
        title: str = "Sample Results",
        *,
        include_min: bool = True,
        max_items: int | None = None,
        scrollable: bool = True,
        detailed_labels: bool = False,
        sort_values: bool = True,
        summary_metric_key: str = "mean",
    ):
        super().__init__(title)
        self._include_min = include_min
        self._max_items = max_items
        self._scrollable = scrollable
        self._detailed_labels = detailed_labels
        self._sort_values = sort_values
        self._summary_metric_key = summary_metric_key
        self._summary_metric_keys: list[str] = []
        self._summary_outputs: list[dict] = []
        self._summary_metrics: list[str] = []

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(6, 6, 6, 6)
        self._content_layout.setSpacing(6)

        self._summary_container = QWidget()
        self._summary_layout = QVBoxLayout(self._summary_container)
        self._summary_layout.setContentsMargins(0, 0, 0, 0)
        self._summary_layout.setSpacing(4)
        self._summary_samples = QLabel("Samples: —")
        self._summary_metric_label = QLabel("Metric:")
        self._summary_metric_combo = QComboBox()
        self._summary_metric_combo.setEnabled(False)
        self._summary_metric_combo.currentIndexChanged.connect(self._on_summary_metric_changed)
        for label in (self._summary_samples, self._summary_metric_label):
            label.setWordWrap(True)
        self._summary_layout.addWidget(self._summary_samples)
        metric_row = QHBoxLayout()
        metric_row.addWidget(self._summary_metric_label)
        metric_row.addWidget(self._summary_metric_combo)
        metric_row.addStretch()
        self._summary_layout.addLayout(metric_row)
        self._summary_detail_widgets: list[QWidget] = []

        self._scroll_content = QWidget()
        self._scroll_content.setMaximumWidth(320)
        self._scroll_layout = QVBoxLayout(self._scroll_content)
        self._scroll_layout.setContentsMargins(0, 0, 0, 0)
        self._scroll_layout.setSpacing(6)
        self._scroll_layout.addWidget(self._summary_container)
        if self._scrollable:
            self._scroll = QScrollArea()
            self._scroll.setObjectName("soft-scroll")
            self._scroll.setWidgetResizable(True)
            self._scroll.setFrameShape(QScrollArea.Shape.NoFrame)
            self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            self._scroll.setMinimumHeight(180)
            self._scroll.setWidget(self._scroll_content)
            self._content_layout.addWidget(self._scroll)
        else:
            self._content_layout.addWidget(self._scroll_content)

        container = QVBoxLayout(self)
        container.setContentsMargins(6, 6, 6, 6)
        container.setSpacing(6)
        container.addWidget(self._content)

        self.set_result(None)

    def set_result(self, result: dict | None):
        self._clear_scroll()
        self._clear_summary_details()
        if not result:
            self._summary_samples.setText("Samples: —")
            self._summary_metric_combo.blockSignals(True)
            self._summary_metric_combo.clear()
            self._summary_metric_combo.blockSignals(False)
            self._summary_metric_combo.setEnabled(False)
            self._summary_metric_keys = []
            self._summary_outputs = []
            self._summary_metrics = []
            placeholder = QLabel("No sample results available.")
            placeholder.setObjectName("label")
            self._scroll_layout.addWidget(placeholder)
            self._scroll_layout.addStretch(1)
            return

        num_samples = result.get("num_samples", 0)
        metrics = list(result.get("metrics", []))
        if not self._include_min:
            metrics = [m for m in metrics if m != "min"]
        outputs = list(result.get("outputs", []))

        metric_map = get_metric_map()
        metric_names = [metric_map[m].name if m in metric_map else m for m in metrics]

        self._summary_samples.setText(f"Samples: {num_samples}")
        self._summary_outputs = outputs
        self._summary_metrics = metrics
        self._summary_metric_keys = metrics[:]

        self._summary_metric_combo.blockSignals(True)
        self._summary_metric_combo.clear()
        for key in metrics:
            label = metric_map.get(key).name if key in metric_map else key
            self._summary_metric_combo.addItem(label, key)
        preferred_key = self._summary_metric_key if self._summary_metric_key in metrics else None
        if preferred_key is None and "mean" in metrics:
            preferred_key = "mean"
        if preferred_key is None and metrics:
            preferred_key = metrics[0]
        if preferred_key in metrics:
            self._summary_metric_combo.setCurrentIndex(metrics.index(preferred_key))
        self._summary_metric_combo.blockSignals(False)
        self._summary_metric_combo.setEnabled(bool(metrics))

        summary_metric_key = preferred_key if preferred_key is not None else (metrics[0] if metrics else "mean")
        self._summary_metric_key = summary_metric_key
        self._build_summary_top_activations(outputs, metrics, metric_map, metric_key=summary_metric_key)

        if self._max_items is not None and not self._detailed_labels:
            return

        layer_counter = 1
        has_named_output = any(
            "output" in (entry.get("name", "") or "").lower() for entry in outputs
        )
        default_output_index = 0 if self._detailed_labels and outputs and not has_named_output else None
        for output_index, output_entry in enumerate(outputs):
            name = output_entry.get("name", "output")
            shape = output_entry.get("shape", [])
            force_output = default_output_index is not None and output_index == default_output_index
            title, used_layer = self._format_output_title(name, shape, layer_counter, force_output=force_output)
            if used_layer:
                layer_counter += 1
            output_group = QGroupBox(title)
            output_layout = QVBoxLayout(output_group)
            output_layout.setContentsMargins(6, 6, 6, 6)
            output_layout.setSpacing(4)

            values = output_entry.get("values", {}) or {}
            for metric_key in metrics:
                metric_title = metric_map.get(metric_key).name if metric_key in metric_map else metric_key
                metric_values = values.get(metric_key, []) or []
                metric_layout = self._build_metric_layout(
                    metric_key,
                    metric_title,
                    metric_values,
                    max_items=self._max_items,
                )
                output_layout.addLayout(metric_layout)

            self._scroll_layout.addWidget(output_group)

        self._scroll_layout.addStretch(1)

    def _clear_summary_details(self):
        for widget in self._summary_detail_widgets:
            self._summary_layout.removeWidget(widget)
            widget.setParent(None)
        self._summary_detail_widgets.clear()

    def _clear_scroll(self):
        summary_widget = self._summary_container
        while self._scroll_layout.count():
            item = self._scroll_layout.takeAt(0)
            widget = item.widget()
            if widget is not None and widget is not summary_widget:
                widget.setParent(None)
        self._scroll_layout.addWidget(summary_widget)

    def _build_metric_layout(
        self,
        metric_key: str,
        metric_title: str,
        values: Iterable[Any],
        *,
        max_items: int | None = None,
    ) -> QGridLayout:
        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(2)

        title = QLabel(f"{metric_title}:")
        title.setObjectName("label")
        title.setWordWrap(True)
        layout.addWidget(title, 0, 0, 1, 2)

        indexed = [(i, float(v)) for i, v in enumerate(values)]
        if self._sort_values:
            reverse = metric_key != "min"
            indexed.sort(key=lambda x: x[1], reverse=reverse)

        if max_items is not None:
            indexed = indexed[:max_items]

        for row, (idx, value) in enumerate(indexed, start=1):
            idx_label = QLabel(self._format_index_label(idx))
            idx_label.setObjectName("label")
            val_label = QLabel(f"{value:.6f}")
            val_label.setObjectName("label")
            val_label.setWordWrap(False)
            val_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            layout.addWidget(idx_label, row, 0)
            layout.addWidget(val_label, row, 1)

        return layout

    def _build_summary_top_activations(
        self,
        outputs: list[dict],
        metrics: list[str],
        metric_map: dict[str, Any],
        *,
        limit: int = 10,
        metric_key: str = "mean",
    ) -> None:
        if not outputs or not metrics:
            return
        if metric_key not in metrics:
            metric_key = metrics[0]

        layer_counter = 1
        has_named_output = any(
            "output" in (entry.get("name", "") or "").lower() for entry in outputs
        )
        default_output_index = 0 if outputs and not has_named_output else None

        labeled_outputs: list[tuple[dict, str]] = []
        for output_index, output_entry in enumerate(outputs):
            name = output_entry.get("name", "output")
            force_output = default_output_index is not None and output_index == default_output_index
            layer_label = self._pretty_layer_label(name, layer_counter, force_output=force_output)
            if layer_label.startswith("Layer "):
                layer_counter += 1
            labeled_outputs.append((output_entry, layer_label))

        hidden_layers = layer_counter - 1
        output_layer_index = hidden_layers + 1

        def resolve_layer_index(layer_label: str) -> int | None:
            if layer_label.startswith("Layer "):
                suffix = layer_label.replace("Layer ", "", 1).strip()
                if suffix.isdigit():
                    return int(suffix)
            lowered = layer_label.lower()
            if lowered == "input":
                return 0
            if lowered == "output":
                return output_layer_index
            return None

        activations: list[tuple[float, str, int, int | None]] = []
        for output_entry, layer_label in labeled_outputs:
            layer_index = resolve_layer_index(layer_label)
            values = output_entry.get("values", {}) or {}
            metric_values = values.get(metric_key, []) or []
            for idx, value in enumerate(metric_values):
                activations.append((float(value), layer_label, idx, layer_index))

        if not activations:
            return

        activations.sort(key=lambda item: item[0], reverse=True)
        top_items = activations[:limit]

        group = QGroupBox("")
        group.setObjectName("summary-box")
        group_layout = QGridLayout(group)
        group_layout.setContentsMargins(6, 6, 6, 6)
        group_layout.setHorizontalSpacing(8)
        group_layout.setVerticalSpacing(2)

        for row, (value, layer_label, idx, layer_index) in enumerate(top_items):
            if layer_index is not None:
                label_text = f"L{layer_index} N{idx}"
            else:
                short_label = self._short_layer_label(layer_label)
                if short_label == "Out":
                    label_text = f"Out {idx}"
                elif short_label == "In":
                    label_text = f"In {idx}"
                else:
                    label_text = f"{short_label} N{idx}"
            idx_label = QLabel(label_text)
            idx_label.setObjectName("label")
            val_label = QLabel(f"{value:.6f}")
            val_label.setObjectName("label")
            val_label.setWordWrap(False)
            val_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            group_layout.addWidget(idx_label, row, 0)
            group_layout.addWidget(val_label, row, 1)

        self._summary_layout.addWidget(group)
        self._summary_detail_widgets.append(group)

    def _on_summary_metric_changed(self, index: int) -> None:
        if index < 0 or index >= len(self._summary_metric_keys):
            return
        metric_key = self._summary_metric_keys[index]
        if not self._summary_outputs or not self._summary_metrics:
            return
        self._summary_metric_key = metric_key
        metric_map = get_metric_map()
        self._clear_summary_details()
        self._build_summary_top_activations(
            self._summary_outputs,
            self._summary_metrics,
            metric_map,
            metric_key=metric_key,
        )

    def _short_layer_label(self, layer_label: str) -> str:
        if layer_label.startswith("Layer "):
            suffix = layer_label.replace("Layer ", "", 1).strip()
            return f"L{suffix}" if suffix else "L"
        lowered = layer_label.lower()
        if lowered == "input":
            return "In"
        if lowered == "output":
            return "Out"
        return "L"

    def _format_index_label(self, idx: int) -> str:
        if self._detailed_labels:
            return f"Neuron {idx}"
        return f"v[{idx}]"

    def _format_output_title(
        self,
        name: str,
        shape: list[Any],
        layer_counter: int,
        *,
        force_output: bool = False,
    ) -> tuple[str, bool]:
        if not self._detailed_labels:
            return f"{name} (shape {shape})", False

        label = self._pretty_layer_label(name, layer_counter, force_output=force_output)
        title = f"{label} (shape {shape})"
        return title, label.startswith("Layer ")

    def _pretty_layer_label(self, name: str, layer_counter: int, *, force_output: bool = False) -> str:
        lowered = (name or "").lower()
        is_input = "input" in lowered
        is_output = "output" in lowered
        if force_output:
            is_output = True
        if is_input and not is_output:
            label = "Input"
        elif is_output:
            label = "Output"
        else:
            label = f"Layer {layer_counter}"

        return label
