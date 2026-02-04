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
    ):
        super().__init__(title)
        self._include_min = include_min
        self._max_items = max_items
        self._scrollable = scrollable

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(6, 6, 6, 6)
        self._content_layout.setSpacing(6)

        self._summary_layout = QVBoxLayout()
        self._summary_layout.setContentsMargins(0, 0, 0, 0)
        self._summary_layout.setSpacing(4)

        self._summary_samples = QLabel("Samples: —")
        for label in (self._summary_samples,):
            label.setWordWrap(True)
        self._summary_layout.addWidget(self._summary_samples)

        self._content_layout.addLayout(self._summary_layout)

        self._scroll_content = QWidget()
        self._scroll_content.setMaximumWidth(320)
        self._scroll_layout = QVBoxLayout(self._scroll_content)
        self._scroll_layout.setContentsMargins(0, 0, 0, 0)
        self._scroll_layout.setSpacing(6)
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
        if not result:
            self._summary_samples.setText("Samples: —")
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

        for output_entry in outputs:
            name = output_entry.get("name", "output")
            shape = output_entry.get("shape", [])
            output_group = QGroupBox(f"{name} (shape {shape})")
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

    def _clear_scroll(self):
        while self._scroll_layout.count():
            item = self._scroll_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

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
        reverse = metric_key != "min"
        indexed.sort(key=lambda x: x[1], reverse=reverse)

        if max_items is not None:
            indexed = indexed[:max_items]

        for row, (idx, value) in enumerate(indexed, start=1):
            idx_label = QLabel(f"v[{idx}]")
            idx_label.setObjectName("label")
            val_label = QLabel(f"{value:.6f}")
            val_label.setObjectName("label")
            val_label.setWordWrap(False)
            val_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            layout.addWidget(idx_label, row, 0)
            layout.addWidget(val_label, row, 1)

        return layout
