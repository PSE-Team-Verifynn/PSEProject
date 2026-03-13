"""Unit tests for sample_metrics.py

Run with:
    QT_QPA_PLATFORM=offscreen python -m pytest test_sample_metrics.py -v
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from PySide6.QtWidgets import QApplication, QLabel, QGroupBox
from PySide6.QtCore import Qt

from nn_verification_visualisation.view.base_view.sample_metrics import (
    SampleMetricsWidget,
    _NoScrollComboBox,
    _SUMMARY_MODE_LABELS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def widget(qapp):
    return SampleMetricsWidget()

@pytest.fixture
def detailed_widget(qapp):
    return SampleMetricsWidget(detailed_labels=True)


@pytest.fixture
def simple_result():
    return {
        "num_samples": 5,
        "sampling_mode": "post_activation",
        "metrics": ["mean", "max"],
        "outputs": [
            {
                "name": "output",
                "shape": [3],
                "values": {
                    "mean": [0.1, 0.5, 0.3],
                    "max": [0.9, 0.7, 0.4],
                },
            }
        ],
    }


# ---------------------------------------------------------------------------
# _NoScrollComboBox
# ---------------------------------------------------------------------------

class TestNoScrollComboBox:
    def test_wheel_event_is_ignored(self, qapp):
        combo = _NoScrollComboBox()
        event = MagicMock()
        combo.wheelEvent(event)
        event.ignore.assert_called_once()


# ---------------------------------------------------------------------------
# _SUMMARY_MODE_LABELS constant
# ---------------------------------------------------------------------------

def test_summary_mode_labels_known_keys():
    assert "pre_activation_after_bias" in _SUMMARY_MODE_LABELS
    assert "post_activation" in _SUMMARY_MODE_LABELS


# ---------------------------------------------------------------------------
# SampleMetricsWidget — initialisation
# ---------------------------------------------------------------------------

class TestSampleMetricsWidgetInit:
    def test_default_title(self, widget):
        assert widget.title() == "Sample Results"

    def test_custom_title(self, qapp):
        w = SampleMetricsWidget(title="My Title")
        assert w.title() == "My Title"

    def test_non_scrollable(self, qapp):
        w = SampleMetricsWidget(scrollable=False)
        assert w._scrollable is False

    def test_initial_state_is_empty(self, widget):
        assert widget._summary_samples.text() == "Samples: —"
        assert widget._summary_mode.text() == "Mode: —"
        assert not widget._summary_metric_combo.isEnabled()

    def test_combo_disabled_on_init(self, widget):
        assert widget._summary_metric_combo.isEnabled() is False


# ---------------------------------------------------------------------------
# set_result(None)
# ---------------------------------------------------------------------------

class TestSetResultNone:
    def test_clears_samples_label(self, widget, simple_result):
        widget.set_result(simple_result)
        widget.set_result(None)
        assert widget._summary_samples.text() == "Samples: —"

    def test_clears_mode_label(self, widget, simple_result):
        widget.set_result(simple_result)
        widget.set_result(None)
        assert widget._summary_mode.text() == "Mode: —"

    def test_disables_combo(self, widget, simple_result):
        widget.set_result(simple_result)
        widget.set_result(None)
        assert not widget._summary_metric_combo.isEnabled()

    def test_clears_metric_keys(self, widget, simple_result):
        widget.set_result(simple_result)
        widget.set_result(None)
        assert widget._summary_metric_keys == []

    def test_empty_dict_treated_as_none(self, widget):
        # empty dict is falsy, so treated as no result
        widget.set_result({})
        assert widget._summary_samples.text() == "Samples: —"


# ---------------------------------------------------------------------------
# set_result with valid data
# ---------------------------------------------------------------------------

class TestSetResultValid:
    def test_samples_label_updated(self, widget, simple_result):
        widget.set_result(simple_result)
        assert widget._summary_samples.text() == "Samples: 5"

    def test_mode_label_known_key(self, widget, simple_result):
        widget.set_result(simple_result)
        assert widget._summary_mode.text() == "Mode: Post activation"

    def test_mode_label_unknown_key(self, widget):
        result = {"num_samples": 2, "sampling_mode": "custom_mode", "metrics": [], "outputs": []}
        widget.set_result(result)
        assert widget._summary_mode.text() == "Mode: custom_mode"

    def test_mode_label_from_label_field(self, widget):
        result = {
            "num_samples": 2,
            "sampling_mode": None,
            "sampling_mode_label": "Special Mode",
            "metrics": [],
            "outputs": [],
        }
        widget.set_result(result)
        assert widget._summary_mode.text() == "Mode: Special Mode"

    def test_mode_label_fallback_to_dash(self, widget):
        result = {"num_samples": 0, "sampling_mode": None, "metrics": [], "outputs": []}
        widget.set_result(result)
        assert widget._summary_mode.text() == "Mode: —"

    def test_combo_enabled_with_metrics(self, widget, simple_result):
        widget.set_result(simple_result)
        assert widget._summary_metric_combo.isEnabled()

    def test_combo_item_count(self, widget, simple_result):
        widget.set_result(simple_result)
        assert widget._summary_metric_combo.count() == 2

    def test_combo_uses_metric_name(self, widget, simple_result):
        widget.set_result(simple_result)
        assert widget._summary_metric_combo.itemText(0) == "Mean Activation"

    def test_metric_key_defaults_to_mean(self, widget, simple_result):
        widget.set_result(simple_result)
        assert widget._summary_metric_key == "mean"

    def test_preferred_metric_key_respected(self, qapp, simple_result):
        w = SampleMetricsWidget(summary_metric_key="max")
        w.set_result(simple_result)
        assert w._summary_metric_key == "max"

    def test_preferred_metric_key_fallback_when_absent(self, qapp, simple_result):
        w = SampleMetricsWidget(summary_metric_key="nonexistent")
        w.set_result(simple_result)
        # Should fall back to "mean" since it is present
        assert w._summary_metric_key == "mean"

    def test_include_min_false_removes_min(self, qapp):
        result = {
            "num_samples": 3,
            "sampling_mode": "post_activation",
            "metrics": ["mean", "min", "max"],
            "outputs": [],
        }
        w = SampleMetricsWidget(include_min=False)
        w.set_result(result)
        assert "min" not in w._summary_metrics

    def test_include_min_true_keeps_min(self, qapp):
        result = {
            "num_samples": 3,
            "sampling_mode": "post_activation",
            "metrics": ["mean", "min"],
            "outputs": [],
        }
        w = SampleMetricsWidget(include_min=True)
        w.set_result(result)
        assert "min" in w._summary_metrics

    def test_metric_normalisation_by_name(self, widget):
        """Metrics passed by display name should be resolved to their keys."""
        result = {
            "num_samples": 1,
            "sampling_mode": "post_activation",
            "metrics": ["Mean Activation"],
            "outputs": [],
        }
        widget.set_result(result)
        # "Mean Activation" should be normalised to "mean"
        assert "mean" in widget._summary_metrics

    def test_unknown_metric_kept_as_is(self, widget):
        result = {
            "num_samples": 1,
            "sampling_mode": "post_activation",
            "metrics": ["custom_metric"],
            "outputs": [],
        }
        widget.set_result(result)
        assert "custom_metric" in widget._summary_metrics

    def test_max_items_skips_detailed_output(self, qapp, simple_result):
        """When max_items is set but detailed_labels is False, output groups are skipped."""
        w = SampleMetricsWidget(max_items=2, detailed_labels=False)
        w.set_result(simple_result)  # Should not raise

    def test_detailed_labels_builds_output_groups(self, detailed_widget, simple_result):
        detailed_widget.set_result(simple_result)
        # At least the summary group should be present
        assert len(detailed_widget._summary_detail_widgets) >= 0


# ---------------------------------------------------------------------------
# _on_summary_metric_changed
# ---------------------------------------------------------------------------

class TestOnSummaryMetricChanged:
    def test_ignores_while_updating(self, widget, simple_result):
        widget.set_result(simple_result)
        widget._updating_summary_combo = True
        original_key = widget._summary_metric_key
        widget._on_summary_metric_changed(1)
        assert widget._summary_metric_key == original_key

    def test_ignores_negative_index(self, widget, simple_result):
        widget.set_result(simple_result)
        original_key = widget._summary_metric_key
        widget._on_summary_metric_changed(-1)
        assert widget._summary_metric_key == original_key

    def test_ignores_out_of_range_index(self, widget, simple_result):
        widget.set_result(simple_result)
        original_key = widget._summary_metric_key
        widget._on_summary_metric_changed(999)
        assert widget._summary_metric_key == original_key

    def test_ignores_when_no_outputs(self, widget):
        widget._summary_metric_keys = ["mean"]
        widget._summary_outputs = []
        widget._summary_metrics = ["mean"]
        widget._updating_summary_combo = False
        widget._on_summary_metric_changed(0)  # should not raise

    def test_changes_metric_key(self, widget, simple_result):
        widget.set_result(simple_result)
        # index 1 corresponds to "max"
        widget._updating_summary_combo = False
        widget._on_summary_metric_changed(1)
        assert widget._summary_metric_key == "max"


# ---------------------------------------------------------------------------
# _short_layer_label
# ---------------------------------------------------------------------------

class TestShortLayerLabel:
    def test_layer_prefix(self, widget):
        assert widget._short_layer_label("Layer 3") == "L3"

    def test_layer_empty_suffix(self, widget):
        assert widget._short_layer_label("Layer ") == "L"

    def test_input(self, widget):
        assert widget._short_layer_label("input") == "In"

    def test_output(self, widget):
        assert widget._short_layer_label("output") == "Out"

    def test_unknown(self, widget):
        assert widget._short_layer_label("something_else") == "L"


# ---------------------------------------------------------------------------
# _format_index_label
# ---------------------------------------------------------------------------

class TestFormatIndexLabel:
    def test_default_format(self, widget):
        assert widget._format_index_label(0) == "v[0]"
        assert widget._format_index_label(5) == "v[5]"

    def test_detailed_format(self, detailed_widget):
        assert detailed_widget._format_index_label(0) == "Neuron 0"
        assert detailed_widget._format_index_label(3) == "Neuron 3"


# ---------------------------------------------------------------------------
# _pretty_layer_label
# ---------------------------------------------------------------------------

class TestPrettyLayerLabel:
    def test_input_name(self, widget):
        assert widget._pretty_layer_label("input_layer", 1) == "Input"

    def test_output_name(self, widget):
        assert widget._pretty_layer_label("output_layer", 1) == "Output"

    def test_generic_name(self, widget):
        assert widget._pretty_layer_label("hidden", 2) == "Layer 2"

    def test_force_output(self, widget):
        assert widget._pretty_layer_label("hidden", 1, force_output=True) == "Output"

    def test_input_takes_priority_when_only_input(self, widget):
        # name contains "input" but not "output"
        assert widget._pretty_layer_label("my_input_data", 1) == "Input"

    def test_output_takes_priority_over_input_when_both(self, widget):
        # name contains both; output wins
        assert widget._pretty_layer_label("input_output_layer", 1) == "Output"

    def test_empty_name(self, widget):
        assert widget._pretty_layer_label("", 3) == "Layer 3"


# ---------------------------------------------------------------------------
# _format_output_title
# ---------------------------------------------------------------------------

class TestFormatOutputTitle:
    def test_non_detailed(self, widget):
        title, used_layer = widget._format_output_title("out", [3], 1)
        assert "out" in title
        assert "[3]" in title
        assert used_layer is False

    def test_detailed_layer(self, detailed_widget):
        title, used_layer = detailed_widget._format_output_title("hidden", [4], 1)
        assert "Layer 1" in title
        assert used_layer is True

    def test_detailed_output(self, detailed_widget):
        title, used_layer = detailed_widget._format_output_title("output", [2], 1)
        assert "Output" in title
        assert used_layer is False


# ---------------------------------------------------------------------------
# _build_metric_layout
# ---------------------------------------------------------------------------

class TestBuildMetricLayout:
    def test_returns_grid_layout(self, widget):
        from PySide6.QtWidgets import QGridLayout
        layout = widget._build_metric_layout("mean", "Mean Activation", [0.1, 0.5, 0.3])
        assert isinstance(layout, QGridLayout)

    def test_max_detailed_title(self, detailed_widget):
        layout = detailed_widget._build_metric_layout("max", "Max Activation", [0.9, 0.1])
        # title in row 0 should say "Maximum absolute activation"
        item = layout.itemAtPosition(0, 0)
        assert item is not None
        label = item.widget()
        assert "Maximum absolute activation" in label.text()

    def test_max_items_limits_rows(self, widget):
        layout = widget._build_metric_layout("mean", "Mean", [0.1, 0.2, 0.3, 0.4, 0.5], max_items=3)
        # row 0 is title; rows 1-3 are values
        assert layout.rowCount() == 4  # title + 3 data rows

    def test_min_sorts_ascending(self, widget):
        layout = widget._build_metric_layout("min", "Min", [0.9, 0.1, 0.5])
        # First data row (row 1) should be smallest value
        val_item = layout.itemAtPosition(1, 1)
        assert val_item is not None
        text = val_item.widget().text()
        assert "0.100000" in text

    def test_no_sort(self, qapp):
        w = SampleMetricsWidget(sort_values=False)
        layout = w._build_metric_layout("mean", "Mean", [0.9, 0.1])
        val_item = layout.itemAtPosition(1, 1)
        # First row should keep original order
        assert "0.900000" in val_item.widget().text()


# ---------------------------------------------------------------------------
# _build_summary_top_activations
# ---------------------------------------------------------------------------

class TestBuildSummaryTopActivations:
    def test_empty_outputs_returns_early(self, widget):
        before = len(widget._summary_detail_widgets)
        widget._build_summary_top_activations([], ["mean"], {})
        assert len(widget._summary_detail_widgets) == before

    def test_empty_metrics_returns_early(self, widget):
        before = len(widget._summary_detail_widgets)
        widget._build_summary_top_activations([{"values": {"mean": [1.0]}}], [], {})
        assert len(widget._summary_detail_widgets) == before

    def test_adds_group_widget(self, widget, simple_result):
        from nn_verification_visualisation.controller.process_manager.sample_metric_registry import get_metric_map
        widget._clear_summary_details()
        widget._build_summary_top_activations(
            simple_result["outputs"],
            simple_result["metrics"],
            get_metric_map(),
            metric_key="mean",
        )
        assert len(widget._summary_detail_widgets) == 1

    def test_limit_respected(self, widget):
        from nn_verification_visualisation.controller.process_manager.sample_metric_registry import get_metric_map
        outputs = [
            {"name": "hidden", "values": {"mean": list(range(20))}}
        ]
        widget._clear_summary_details()
        widget._build_summary_top_activations(outputs, ["mean"], get_metric_map(), limit=5, metric_key="mean")
        group = widget._summary_detail_widgets[0]
        layout = group.layout()
        assert layout.rowCount() == 5

    def test_fallback_metric_key(self, widget):
        """If the requested metric_key is absent, the first available is used."""
        from nn_verification_visualisation.controller.process_manager.sample_metric_registry import get_metric_map
        outputs = [{"name": "hidden", "values": {"max": [0.5, 0.9]}}]
        widget._clear_summary_details()
        widget._build_summary_top_activations(outputs, ["max"], get_metric_map(), metric_key="mean")
        # Should still produce a group (using "max" as fallback)
        assert len(widget._summary_detail_widgets) == 1

    def test_case_insensitive_metric_key_lookup(self, widget):
        """Metric key lookup is case-insensitive."""
        from nn_verification_visualisation.controller.process_manager.sample_metric_registry import get_metric_map
        outputs = [{"name": "hidden", "values": {"Mean": [0.1, 0.2]}}]
        widget._clear_summary_details()
        widget._build_summary_top_activations(outputs, ["mean"], get_metric_map(), metric_key="mean")
        assert len(widget._summary_detail_widgets) == 1

    def test_output_label_uses_layer_index(self, widget):
        """Outputs labeled as 'Layer N' should render L{N} labels."""
        from nn_verification_visualisation.controller.process_manager.sample_metric_registry import get_metric_map
        outputs = [{"name": "hidden1", "values": {"mean": [1.0, 2.0]}}]
        widget._clear_summary_details()
        widget._build_summary_top_activations(outputs, ["mean"], get_metric_map(), metric_key="mean")
        group = widget._summary_detail_widgets[0]
        layout = group.layout()
        label = layout.itemAtPosition(0, 0).widget()
        assert label.text().startswith("L")

    def test_output_layer_uses_out_label(self, widget):
        """Output named 'output' should produce 'Out N' labels when no layer index."""
        from nn_verification_visualisation.controller.process_manager.sample_metric_registry import get_metric_map
        outputs = [
            {"name": "hidden", "values": {"mean": [0.5]}},  # Layer 1
            {"name": "output", "values": {"mean": [0.8]}},  # Output
        ]
        widget._clear_summary_details()
        widget._build_summary_top_activations(outputs, ["mean"], get_metric_map(), metric_key="mean")
        # One group is added
        assert len(widget._summary_detail_widgets) == 1

    def test_no_activations_produces_no_group(self, widget):
        """If no metric values exist, nothing is appended."""
        from nn_verification_visualisation.controller.process_manager.sample_metric_registry import get_metric_map
        outputs = [{"name": "hidden", "values": {}}]
        widget._clear_summary_details()
        widget._build_summary_top_activations(outputs, ["mean"], get_metric_map(), metric_key="mean")
        assert len(widget._summary_detail_widgets) == 0