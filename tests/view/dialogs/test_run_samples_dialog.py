from __future__ import annotations

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication

from nn_verification_visualisation.view.dialogs.run_samples_dialog import (
    RunSamplesDialog,
    _SampleWorker,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bounds(values=None):
    b = Mock()
    b.get_values.return_value = values or [(0.0, 1.0)]
    return b


def _make_config(bounds=None, selected_index=0):
    cfg = Mock()
    cfg.saved_bounds = bounds if bounds is not None else [_make_bounds()]
    cfg.selected_bounds_index = selected_index
    return cfg


def _make_dialog(qtbot, mocker, config=None, on_results=None):
    """Patch heavy imports and return a RunSamplesDialog."""
    mocker.patch(
        "nn_verification_visualisation.view.dialogs.run_samples_dialog.load_metrics",
        return_value=[],
    )
    mocker.patch(
        "nn_verification_visualisation.view.dialogs.run_samples_dialog.Storage",
    )
    cfg = config or _make_config()
    dlg = RunSamplesDialog(on_close=Mock(), config=cfg, on_results=on_results)
    qtbot.addWidget(dlg)
    return dlg


# ---------------------------------------------------------------------------
# _SampleWorker
# ---------------------------------------------------------------------------

class TestSampleWorker:
    def test_run_emits_finished_on_success(self, qtbot):
        cfg = Mock()
        with patch(
            "nn_verification_visualisation.view.dialogs.run_samples_dialog.run_samples_for_bounds",
            return_value={"key": "value"},
        ):
            worker = _SampleWorker(cfg, [(0, 1)], 10, ["metric"], "mode")
            results = []
            worker.finished.connect(results.append)
            worker.run()
        assert results == [{"key": "value"}]

    def test_run_emits_failed_on_exception(self, qtbot):
        cfg = Mock()
        with patch(
            "nn_verification_visualisation.view.dialogs.run_samples_dialog.run_samples_for_bounds",
            side_effect=RuntimeError("boom"),
        ):
            worker = _SampleWorker(cfg, [(0, 1)], 10, ["metric"], "mode")
            errors = []
            worker.failed.connect(errors.append)
            worker.run()
        assert errors == ["boom"]


# ---------------------------------------------------------------------------
# RunSamplesDialog – initialisation
# ---------------------------------------------------------------------------

class TestRunSamplesDialogInit:
    def test_run_button_disabled_when_no_bounds(self, qtbot, mocker):
        mocker.patch(
            "nn_verification_visualisation.view.dialogs.run_samples_dialog.load_metrics",
            return_value=[],
        )
        mocker.patch("nn_verification_visualisation.view.dialogs.run_samples_dialog.Storage")
        cfg = _make_config(bounds=[])
        dlg = RunSamplesDialog(on_close=Mock(), config=cfg)
        qtbot.addWidget(dlg)

        assert not dlg._run_button.isEnabled()
        assert "No saved bounds" in dlg._status_label.text()

    def test_run_button_enabled_when_bounds_exist(self, qtbot, mocker):
        dlg = _make_dialog(qtbot, mocker)
        assert dlg._run_button.isEnabled()

    def test_bounds_selector_populated(self, qtbot, mocker):
        mocker.patch(
            "nn_verification_visualisation.view.dialogs.run_samples_dialog.load_metrics",
            return_value=[],
        )
        mocker.patch("nn_verification_visualisation.view.dialogs.run_samples_dialog.Storage")
        cfg = _make_config(bounds=[_make_bounds(), _make_bounds()], selected_index=1)
        dlg = RunSamplesDialog(on_close=Mock(), config=cfg)
        qtbot.addWidget(dlg)

        assert dlg._bounds_selector.count() == 2
        assert dlg._bounds_selector.currentIndex() == 1

    def test_metric_checkboxes_created(self, qtbot, mocker):
        metric1 = Mock(name="m1", key="k1")
        metric1.name = "Metric One"
        metric1.key = "k1"
        metric2 = Mock()
        metric2.name = "Metric Two"
        metric2.key = "k2"
        mocker.patch(
            "nn_verification_visualisation.view.dialogs.run_samples_dialog.load_metrics",
            return_value=[metric1, metric2],
        )
        mocker.patch("nn_verification_visualisation.view.dialogs.run_samples_dialog.Storage")
        cfg = _make_config()
        dlg = RunSamplesDialog(on_close=Mock(), config=cfg)
        qtbot.addWidget(dlg)

        assert set(dlg._metric_checks.keys()) == {"k1", "k2"}
        # first two metrics default-checked
        assert dlg._metric_checks["k1"].isChecked()
        assert dlg._metric_checks["k2"].isChecked()


# ---------------------------------------------------------------------------
# __on_run_clicked edge cases
# ---------------------------------------------------------------------------

class TestOnRunClicked:
    def test_no_metrics_selected_shows_status(self, qtbot, mocker):
        dlg = _make_dialog(qtbot, mocker)
        # ensure no metric checkboxes (already empty from mock), but simulate
        # a state where none are checked
        dlg._metric_checks = {}  # no metrics
        dlg._RunSamplesDialog__on_run_clicked()
        assert "metric" in dlg._status_label.text().lower()

    def test_run_clicked_no_bounds_selected_shows_status(self, qtbot, mocker):
        mocker.patch(
            "nn_verification_visualisation.view.dialogs.run_samples_dialog.load_metrics",
            return_value=[],
        )
        mocker.patch("nn_verification_visualisation.view.dialogs.run_samples_dialog.Storage")
        cfg = _make_config(bounds=[])
        dlg = RunSamplesDialog(on_close=Mock(), config=cfg)
        qtbot.addWidget(dlg)

        # Force-enable the button to exercise the code path
        dlg._run_button.setEnabled(True)
        dlg._RunSamplesDialog__on_run_clicked()

        assert "bounds" in dlg._status_label.text().lower()

    def test_valid_run_starts_thread(self, qtbot, mocker):
        mock_cb = Mock()
        mock_cb.isChecked.return_value = True
        dlg = _make_dialog(qtbot, mocker)
        dlg._metric_checks = {"m1": mock_cb}

        # Mock both QThread *and* _SampleWorker so that moveToThread is never
        # called on a real QObject with a Mock argument (PySide6 rejects that).
        mock_thread = Mock()
        mocker.patch(
            "nn_verification_visualisation.view.dialogs.run_samples_dialog.QThread",
            return_value=mock_thread,
        )
        mock_worker = Mock()
        mocker.patch(
            "nn_verification_visualisation.view.dialogs.run_samples_dialog._SampleWorker",
            return_value=mock_worker,
        )

        dlg._RunSamplesDialog__on_run_clicked()

        mock_thread.start.assert_called_once()
        mock_worker.moveToThread.assert_called_once_with(mock_thread)


# ---------------------------------------------------------------------------
# __on_worker_finished / __on_worker_failed
# ---------------------------------------------------------------------------

class TestWorkerCallbacks:
    def test_on_worker_finished_saves_sample(self, qtbot, mocker):
        mock_storage = mocker.patch(
            "nn_verification_visualisation.view.dialogs.run_samples_dialog.Storage"
        )
        mocker.patch(
            "nn_verification_visualisation.view.dialogs.run_samples_dialog.load_metrics",
            return_value=[],
        )
        bounds = _make_bounds()
        cfg = _make_config(bounds=[bounds])
        dlg = RunSamplesDialog(on_close=Mock(), config=cfg)
        qtbot.addWidget(dlg)

        mocker.patch.object(dlg, "_RunSamplesDialog__show_results")
        dlg._RunSamplesDialog__on_worker_finished({"result": 1})

        bounds.set_sample.assert_called_once_with({"result": 1})
        mock_storage.return_value.request_autosave.assert_called_once()
        assert "computed" in dlg._status_label.text().lower()

    def test_on_worker_finished_no_bounds_skips_save(self, qtbot, mocker):
        mocker.patch("nn_verification_visualisation.view.dialogs.run_samples_dialog.Storage")
        mocker.patch(
            "nn_verification_visualisation.view.dialogs.run_samples_dialog.load_metrics",
            return_value=[],
        )
        cfg = _make_config(bounds=[])
        dlg = RunSamplesDialog(on_close=Mock(), config=cfg)
        qtbot.addWidget(dlg)

        mocker.patch.object(dlg, "_RunSamplesDialog__show_results")
        # Should not raise
        dlg._RunSamplesDialog__on_worker_finished({})

    def test_on_worker_failed_shows_error(self, qtbot, mocker):
        dlg = _make_dialog(qtbot, mocker)
        mocker.patch.object(dlg, "_RunSamplesDialog__show_error")

        dlg._RunSamplesDialog__on_worker_failed("something went wrong")

        assert "something went wrong" in dlg._status_label.text()
        dlg._RunSamplesDialog__show_error.assert_called_once()


# ---------------------------------------------------------------------------
# __format_error_message
# ---------------------------------------------------------------------------

class TestFormatErrorMessage:
    def _fmt(self, dlg, msg):
        return dlg._RunSamplesDialog__format_error_message(msg)

    def test_empty_message_returns_default(self, qtbot, mocker):
        dlg = _make_dialog(qtbot, mocker)
        assert self._fmt(dlg, "") == "Failed to run samples."
        assert self._fmt(dlg, None) == "Failed to run samples."

    def test_unsupported_ir_version_truncated(self, qtbot, mocker):
        dlg = _make_dialog(qtbot, mocker)
        msg = "prefix junk Unsupported model IR version details here"
        result = self._fmt(dlg, msg)
        assert result.startswith("Unsupported model IR version")
        assert "prefix junk" not in result

    def test_other_message_returned_as_is(self, qtbot, mocker):
        dlg = _make_dialog(qtbot, mocker)
        assert self._fmt(dlg, "some error") == "some error"


# ---------------------------------------------------------------------------
# __show_results / __show_error
# ---------------------------------------------------------------------------

class TestShowResults:
    def test_show_results_no_parent_calls_on_close(self, qtbot, mocker):
        on_close = Mock()
        dlg = _make_dialog(qtbot, mocker)
        dlg._RunSamplesDialog__on_close = on_close
        # No parent → on_close called, no timer
        mocker.patch.object(dlg, "on_close")
        dlg._RunSamplesDialog__show_results({})
        dlg.on_close.assert_called()

    def test_show_results_with_parent_calls_on_results(self, qtbot, mocker, qapp):
        on_results = Mock()
        dlg = _make_dialog(qtbot, mocker, on_results=on_results)

        parent = Mock()
        parent.open_dialog = Mock()
        parent.close_dialog = Mock()
        mocker.patch.object(dlg, "parent", return_value=parent)
        mocker.patch.object(dlg, "on_close")

        mocker.patch(
            "nn_verification_visualisation.view.dialogs.run_samples_dialog.SampleResultsDialog"
        )
        mocker.patch(
            "nn_verification_visualisation.view.dialogs.run_samples_dialog.QTimer"
        )

        dlg._RunSamplesDialog__show_results({"r": 1})
        dlg.on_close.assert_called_once()

    def test_show_error_no_parent_does_nothing(self, qtbot, mocker):
        dlg = _make_dialog(qtbot, mocker)
        mocker.patch.object(dlg, "parent", return_value=None)
        # Should not raise
        dlg._RunSamplesDialog__show_error("oops")

    def test_show_error_with_parent_queues_timer(self, qtbot, mocker):
        dlg = _make_dialog(qtbot, mocker)
        parent = Mock(spec=["open_dialog", "close_dialog"])
        mocker.patch.object(dlg, "parent", return_value=parent)

        mock_timer = mocker.patch(
            "nn_verification_visualisation.view.dialogs.run_samples_dialog.QTimer"
        )
        dlg._RunSamplesDialog__show_error("error msg")
        mock_timer.singleShot.assert_called_once()


# ---------------------------------------------------------------------------
# closeEvent
# ---------------------------------------------------------------------------

class TestCloseEvent:
    def test_close_event_sets_allow_results_false_when_thread_running(self, qtbot, mocker):
        dlg = _make_dialog(qtbot, mocker)
        mock_thread = Mock()
        mock_thread.isRunning.return_value = True
        dlg._thread = mock_thread

        event = Mock(spec=QEvent)
        with patch.object(type(dlg).__bases__[0], "closeEvent", return_value=None):
            dlg.closeEvent(event)

        assert dlg._allow_results is False

    def test_close_event_leaves_allow_results_when_no_thread(self, qtbot, mocker):
        dlg = _make_dialog(qtbot, mocker)
        dlg._thread = None

        event = Mock(spec=QEvent)
        with patch.object(type(dlg).__bases__[0], "closeEvent", return_value=None):
            dlg.closeEvent(event)

        assert dlg._allow_results is True

    def test_on_thread_finished_clears_references(self, qtbot, mocker):
        dlg = _make_dialog(qtbot, mocker)
        dlg._thread = Mock()
        dlg._worker = Mock()

        dlg._RunSamplesDialog__on_thread_finished()

        assert dlg._thread is None
        assert dlg._worker is None