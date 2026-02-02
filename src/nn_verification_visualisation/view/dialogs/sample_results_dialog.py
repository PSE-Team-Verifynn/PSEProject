from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QTextOption
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QGroupBox, QPlainTextEdit

from nn_verification_visualisation.controller.process_manager.sample_metric_registry import get_metric_map
from nn_verification_visualisation.view.dialogs.dialog_base import DialogBase


class SampleResultsDialog(DialogBase):
    def __init__(self, on_close: Callable[[], None], result: dict):
        self.result = result
        super().__init__(on_close, "Sample Results", (560, 420))

    def get_content(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        summary_group = QGroupBox("Summary")
        summary_layout = QVBoxLayout(summary_group)
        summary_layout.setContentsMargins(8, 8, 8, 8)

        num_samples = self.result.get("num_samples", 0)
        metrics = self.result.get("metrics", [])
        outputs = self.result.get("outputs", [])
        metric_map = get_metric_map()

        def metric_label(key: str) -> str:
            metric = metric_map.get(key)
            return metric.name if metric else key

        summary_layout.addWidget(QLabel(f"Samples: {num_samples}"))
        summary_layout.addWidget(QLabel(f"Metrics: {', '.join(metric_label(m) for m in metrics)}"))
        summary_layout.addWidget(QLabel(f"Outputs sampled: {len(outputs)}"))

        layout.addWidget(summary_group)

        text = QPlainTextEdit()
        text.setReadOnly(True)
        text.setObjectName("code-block")
        text_lines = []
        for output_entry in outputs:
            name = output_entry.get("name", "output")
            shape = output_entry.get("shape", [])
            text_lines.append(f"{name} (shape {shape}):")
            values = output_entry.get("values", {})
            for metric_key in metrics:
                text_lines.append(f"  {metric_label(metric_key)}:")
                metric_values = values.get(metric_key, [])
                for i, value in enumerate(metric_values):
                    text_lines.append(f"    v[{i}]: {value:.6f}")
            text_lines.append("")
        text.setPlainText("\n".join(text_lines).strip())
        text.setMinimumHeight(220)
        text.setWordWrapMode(QTextOption.WrapMode.NoWrap)
        layout.addWidget(text)

        return container
