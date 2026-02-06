from __future__ import annotations

from typing import Callable, Iterable

from PySide6.QtCore import Qt, QObject, QThread, Signal, QTimer
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QSpinBox,
    QCheckBox,
    QPushButton,
    QGroupBox,
    QComboBox,
    QProgressBar,
)

from nn_verification_visualisation.controller.process_manager.sample_runner import (
    MAX_SAMPLES_PER_RUN,
    DEFAULT_SAMPLING_MODE,
    SAMPLING_MODE_LABELS,
    run_samples_for_bounds,
)
from nn_verification_visualisation.controller.process_manager.sample_metric_registry import load_metrics
from nn_verification_visualisation.model.data.input_bounds import InputBounds
from nn_verification_visualisation.model.data.network_verification_config import NetworkVerificationConfig
from nn_verification_visualisation.view.dialogs.dialog_base import DialogBase
from nn_verification_visualisation.view.dialogs.info_popup import InfoPopup
from nn_verification_visualisation.view.dialogs.info_type import InfoType
from nn_verification_visualisation.view.dialogs.sample_results_dialog import SampleResultsDialog
from nn_verification_visualisation.model.data.storage import Storage


class _SampleWorker(QObject):
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(
        self,
        config: NetworkVerificationConfig,
        bounds: list[tuple[float, float]],
        num_samples: int,
        metrics: Iterable[str],
        sampling_mode: str,
    ):
        super().__init__()
        self._config = config
        self._bounds = bounds
        self._num_samples = num_samples
        self._metrics = list(metrics)
        self._sampling_mode = sampling_mode

    def run(self):
        try:
            result = run_samples_for_bounds(
                self._config.network,
                self._bounds,
                self._num_samples,
                self._metrics,
                self._sampling_mode,
            )
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class RunSamplesDialog(DialogBase):
    config: NetworkVerificationConfig

    def __init__(
        self,
        on_close: Callable[[], None],
        config: NetworkVerificationConfig,
        on_results: Callable[[dict], None] | None = None,
    ):
        self.config = config
        self._on_results = on_results

        self._thread: QThread | None = None
        self._worker: _SampleWorker | None = None
        self._allow_results = True

        self._metric_checks: dict[str, QCheckBox] = {}
        self._status_label = QLabel("")
        self._progress = QProgressBar()
        self._run_button = QPushButton("Run Samples")
        self._cancel_button = QPushButton("Cancel")
        self._bounds_selector = QComboBox()
        self._mode_selector = QComboBox()

        super().__init__(on_close, "Run Samples", (520, 320))

    def get_content(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        info = QLabel("Samples are generated from the currently selected bounds.")
        info.setWordWrap(True)
        layout.addWidget(info)

        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout(settings_group)
        settings_layout.setContentsMargins(8, 8, 8, 8)
        settings_layout.setSpacing(6)

        bounds_row = QHBoxLayout()
        bounds_row.addWidget(QLabel("Bounds:"))
        self.__populate_bounds_selector()
        bounds_row.addWidget(self._bounds_selector)
        bounds_row.addStretch()
        settings_layout.addLayout(bounds_row)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Activation mode:"))
        for mode_key, mode_label in SAMPLING_MODE_LABELS.items():
            self._mode_selector.addItem(mode_label, mode_key)
        self._mode_selector.setMinimumContentsLength(0)
        self._mode_selector.setMinimumWidth(0)
        self._mode_selector.setFixedWidth(200)
        default_mode_index = self._mode_selector.findData(DEFAULT_SAMPLING_MODE)
        if default_mode_index >= 0:
            self._mode_selector.setCurrentIndex(default_mode_index)
        mode_row.addWidget(self._mode_selector)
        mode_row.addStretch()
        settings_layout.addLayout(mode_row)

        sample_row = QHBoxLayout()
        sample_row.addWidget(QLabel("Number of samples:"))
        self._samples_spin = QSpinBox()
        self._samples_spin.setMinimum(1)
        self._samples_spin.setMaximum(MAX_SAMPLES_PER_RUN)
        self._samples_spin.setValue(100)
        sample_row.addWidget(self._samples_spin)
        sample_row.addStretch()
        settings_layout.addLayout(sample_row)

        metrics_label = QLabel("Metrics:")
        settings_layout.addWidget(metrics_label)

        metrics = load_metrics()
        for idx, metric in enumerate(metrics):
            checkbox = QCheckBox(metric.name)
            checkbox.setChecked(idx < 2)
            settings_layout.addWidget(checkbox)
            self._metric_checks[metric.key] = checkbox

        layout.addWidget(settings_group)

        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setObjectName("popup-content")
        layout.addWidget(self._status_label)

        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        button_row = QHBoxLayout()
        button_row.addStretch()
        self._cancel_button.clicked.connect(self.on_close)
        self._run_button.clicked.connect(self.__on_run_clicked)
        button_row.addWidget(self._cancel_button)
        button_row.addWidget(self._run_button)
        layout.addLayout(button_row)

        if self._bounds_selector.count() == 0:
            self._run_button.setEnabled(False)
            self.__set_status("No saved bounds available.")

        return container

    def __on_run_clicked(self):
        bounds = self.__get_selected_bounds()
        if bounds is None:
            self.__set_status("Select a saved bounds set before running samples.")
            return
        self.config.selected_bounds_index = self._bounds_selector.currentIndex()

        metrics = [key for key, cb in self._metric_checks.items() if cb.isChecked()]
        if not metrics:
            self.__set_status("Select at least one metric.")
            return

        num_samples = self._samples_spin.value()
        sampling_mode = self._mode_selector.currentData()

        self.__set_running_state(True)

        self._thread = QThread()
        self._worker = _SampleWorker(
            self.config,
            bounds.get_values(),
            num_samples,
            metrics,
            sampling_mode,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self.__on_worker_finished)
        self._worker.failed.connect(self.__on_worker_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self.__on_thread_finished)
        self._thread.start()

    def __get_selected_bounds(self) -> InputBounds | None:
        index = self._bounds_selector.currentIndex()
        if index < 0 or index >= len(self.config.saved_bounds):
            return None
        return self.config.saved_bounds[index]

    def __populate_bounds_selector(self):
        self._bounds_selector.clear()
        for i, _bounds in enumerate(self.config.saved_bounds):
            self._bounds_selector.addItem(f"Bounds {i + 1:02d}")
        index = self.config.selected_bounds_index
        if 0 <= index < self._bounds_selector.count():
            self._bounds_selector.setCurrentIndex(index)

    def __on_worker_finished(self, result: dict):
        bounds = self.__get_selected_bounds()
        if bounds is not None:
            bounds.set_sample(result)
            Storage().request_autosave()
        self.__set_running_state(False)
        self.__set_status("Samples computed and saved.")
        self.__show_results(result)

    def __on_worker_failed(self, message: str):
        self.__set_running_state(False)
        error_message = self.__format_error_message(message)
        self.__set_status(error_message)
        self.__show_error(error_message)

    def __set_running_state(self, running: bool):
        self._progress.setVisible(running)
        self._run_button.setEnabled(not running)
        self._cancel_button.setEnabled(not running)
        self._samples_spin.setEnabled(not running)
        self._bounds_selector.setEnabled(not running)
        self._mode_selector.setEnabled(not running)
        for checkbox in self._metric_checks.values():
            checkbox.setEnabled(not running)
        self._status_label.setText("Running samples..." if running else "")
    
    def __set_status(self, text: str):
        self._status_label.setText(text)

    def __format_error_message(self, message: str | None) -> str:
        if not message:
            return "Failed to run samples."

        key = "Unsupported model IR version"
        idx = message.find(key)
        if idx != -1:
            return message[idx:].strip()

        return message

    def __show_results(self, result: dict):
        parent = self.parent()
        if parent is None or not hasattr(parent, "open_dialog"):
            self.on_close()
            return
        self.on_close()
        def _open_results():
            if not self._allow_results:
                return
            if parent is None or not hasattr(parent, "open_dialog"):
                return
            if self._on_results is not None:
                self._on_results(result)
            results_dialog = SampleResultsDialog(parent.close_dialog, result)
            parent.open_dialog(results_dialog)

        QTimer.singleShot(0, _open_results)

    def __show_error(self, message: str):
        parent = self.parent()
        if parent is None or not hasattr(parent, "open_dialog"):
            return

        def _open_error():
            if parent is None or not hasattr(parent, "open_dialog"):
                return
            parent.open_dialog(InfoPopup(parent.close_dialog, message, InfoType.ERROR))

        QTimer.singleShot(0, _open_error)

    def __on_thread_finished(self):
        self._thread = None
        self._worker = None

    def closeEvent(self, event):
        if self._thread is not None and self._thread.isRunning():
            self._allow_results = False
        super().closeEvent(event)
