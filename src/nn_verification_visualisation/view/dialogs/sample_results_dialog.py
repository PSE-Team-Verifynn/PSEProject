from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout

from nn_verification_visualisation.view.dialogs.dialog_base import DialogBase
from nn_verification_visualisation.view.widgets.sample_metrics_widget import SampleMetricsWidget


class SampleResultsDialog(DialogBase):
    def __init__(self, on_close: Callable[[], None], result: dict):
        self.result = result
        super().__init__(on_close, "Sample Results", (560, 420))

    def get_content(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        metrics_widget = SampleMetricsWidget(
            "Sample Results",
            detailed_labels=True,
            sort_values=False,
        )
        metrics_widget.set_result(self.result)
        layout.addWidget(metrics_widget)

        return container
