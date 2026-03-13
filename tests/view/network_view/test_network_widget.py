import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QColor, QWheelEvent, QKeyEvent
from PySide6.QtWidgets import QApplication, QSlider, QComboBox
from PySide6.QtTest import QTest

from nn_verification_visualisation.view.network_view.network_widget import NetworkWidget
from nn_verification_visualisation.view.network_view.network_node_representation import NetworkNode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(layers: list[int], name: str = "TestNet") -> Mock:
    """Build a minimal NetworkVerificationConfig mock."""
    config = Mock()
    config.network.name = name
    config.network.model = Mock()
    config.network.model.graph.initializer = []
    config.network.model.graph.node = []
    config.layers_dimensions = layers
    return config


@pytest.fixture
def small_config():
    """3-layer network with < 500 edges → weighted mode."""
    return make_config([4, 5, 3])  # 4*5 + 5*3 = 35 edges


@pytest.fixture
def medium_config():
    """Network with 500-10000 edges → normal mode."""
    return make_config([25, 25, 25])


@pytest.fixture
def large_config():
    """Network with > 10000 edges → performance mode."""
    # 100*101 + 101*100 = 20200 edges
    return make_config([100, 101, 100])


@pytest.fixture
def widget_small(qapp, small_config, qtbot):
    w = NetworkWidget(small_config)
    qtbot.addWidget(w)
    return w


@pytest.fixture
def widget_medium(qapp, medium_config, qtbot):
    w = NetworkWidget(medium_config)
    qtbot.addWidget(w)
    return w


@pytest.fixture
def widget_large(qapp, large_config, qtbot):
    w = NetworkWidget(large_config)
    qtbot.addWidget(w)
    return w


# ---------------------------------------------------------------------------
# Init / Build
# ---------------------------------------------------------------------------

class TestInit:
    def test_node_layers_populated(self, widget_small, small_config):
        assert len(widget_small.node_layers) == len(small_config.layers_dimensions)

    def test_node_counts_match_config(self, widget_small, small_config):
        for i, count in enumerate(small_config.layers_dimensions):
            assert len(widget_small.node_layers[i]) == count

    def test_empty_layers_does_not_crash(self, qapp, qtbot):
        config = make_config([])
        w = NetworkWidget(config)
        qtbot.addWidget(w)
        assert w.node_layers == []

    def test_weighted_mode_for_small_network(self, widget_small):
        assert widget_small.use_weighted_mode is True
        assert widget_small.use_performance_mode is False

    def test_normal_mode_for_medium_network(self, widget_medium):
        assert widget_medium.use_performance_mode is False
        assert widget_medium.use_weighted_mode is False

    def test_performance_mode_for_large_network(self, widget_large):
        assert widget_large.use_performance_mode is True
        assert widget_large.use_weighted_mode is False

    def test_nodes_selectable_default_false(self, widget_small):
        assert widget_small.nodes_selectable is False

    def test_nodes_selectable_can_be_set(self, qapp, small_config, qtbot):
        w = NetworkWidget(small_config, nodes_selectable=True)
        qtbot.addWidget(w)
        assert w.nodes_selectable is True


# ---------------------------------------------------------------------------
# get_height_to_width_changer
# ---------------------------------------------------------------------------

class TestHeightToWidthChanger:
    def test_returns_slider(self, widget_small):
        slider = widget_small.get_height_to_width_changer()
        assert isinstance(slider, QSlider)

    def test_slider_initial_value_matches_ratio(self, widget_small):
        slider = widget_small.get_height_to_width_changer()
        assert slider.value() == int(widget_small.height_to_width_ration * 10)

    def test_slider_change_rebuilds_network(self, widget_small, small_config):
        slider = widget_small.get_height_to_width_changer()
        slider.setValue(15)
        assert widget_small.height_to_width_ration == pytest.approx(1.5)
        assert len(widget_small.node_layers) == len(small_config.layers_dimensions)


# ---------------------------------------------------------------------------
# get_performance_mode_changer
# ---------------------------------------------------------------------------

class TestPerformanceModeChanger:
    def test_returns_combobox(self, widget_small):
        dropdown = widget_small.get_performance_mode_changer()
        assert isinstance(dropdown, QComboBox)

    def test_selecting_performance_mode(self, widget_small):
        dropdown = widget_small.get_performance_mode_changer()
        dropdown.setCurrentIndex(0)
        assert widget_small.use_performance_mode is True
        assert widget_small.use_weighted_mode is False

    def test_selecting_weighted_mode(self, widget_small):
        dropdown = widget_small.get_performance_mode_changer()
        dropdown.setCurrentIndex(1)
        assert widget_small.use_weighted_mode is True
        assert widget_small.use_performance_mode is False

    def test_selecting_normal_mode(self, widget_small):
        dropdown = widget_small.get_performance_mode_changer()
        dropdown.setCurrentIndex(2)
        assert widget_small.use_performance_mode is False
        assert widget_small.use_weighted_mode is False

    def test_mode_change_sets_manual_override(self, widget_small):
        widget_small.get_performance_mode_changer().setCurrentIndex(0)
        assert widget_small.manual_mode_override is True


# ---------------------------------------------------------------------------
# _on_node_clicked
# ---------------------------------------------------------------------------

class TestOnNodeClicked:
    def test_does_nothing_when_not_selectable(self, widget_small):
        callback = Mock()
        widget_small.on_selection_changed = callback
        widget_small._on_node_clicked((0, 0))
        callback.assert_not_called()

    def test_calls_callback_when_selectable(self, qapp, small_config, qtbot):
        callback = Mock(return_value=None)
        w = NetworkWidget(small_config, nodes_selectable=True, on_selection_changed=callback)
        qtbot.addWidget(w)
        w._on_node_clicked((0, 0))
        callback.assert_called_once_with(0, 0)

    def test_updates_node_brush_when_color_returned(self, qapp, small_config, qtbot):
        new_color = QColor("red")
        callback = Mock(return_value=new_color)
        w = NetworkWidget(small_config, nodes_selectable=True, on_selection_changed=callback)
        qtbot.addWidget(w)
        w._on_node_clicked((0, 0))
        # The brush should now be the returned color
        assert w.node_layers[0][0].brush().color() == new_color

    def test_no_brush_update_when_color_is_none(self, qapp, small_config, qtbot):
        callback = Mock(return_value=None)
        w = NetworkWidget(small_config, nodes_selectable=True, on_selection_changed=callback)
        qtbot.addWidget(w)
        original_brush = w.node_layers[0][0].brush().color()
        w._on_node_clicked((0, 0))
        assert w.node_layers[0][0].brush().color() == original_brush


# ---------------------------------------------------------------------------
# select_node / unselect_node
# ---------------------------------------------------------------------------

class TestSelectUnselect:
    def test_select_node_sets_brush(self, widget_small):
        color = QColor("blue")
        widget_small.select_node(0, 0, color)
        assert widget_small.node_layers[0][0].brush().color() == color

    def test_unselect_node_restores_default(self, widget_small):
        widget_small.select_node(0, 0, QColor("blue"))
        widget_small.unselect_node(0, 0)
        assert widget_small.node_layers[0][0].brush().color() == NetworkNode.color_unselected


# ---------------------------------------------------------------------------
# get_weights_from_onnx
# ---------------------------------------------------------------------------

class TestGetWeightsFromOnnx:
    def _make_model_proto(self, weight_arrays: list):
        """Build a minimal ModelProto mock with Gemm nodes."""
        import onnx
        from onnx import numpy_helper

        model = onnx.helper.make_model(
            onnx.helper.make_graph([], "g", [], [])
        )
        graph = model.graph

        for i, arr in enumerate(weight_arrays):
            tensor = numpy_helper.from_array(arr.astype(np.float32), name=f"w{i}")
            graph.initializer.append(tensor)

            node = onnx.helper.make_node(
                "Gemm",
                inputs=[f"in{i}", f"w{i}"],
                outputs=[f"out{i}"],
            )
            graph.node.append(node)

        return model

    def test_extracts_gemm_weights(self, widget_small):
        w1 = np.random.randn(5, 4).astype(np.float32)
        w2 = np.random.randn(3, 5).astype(np.float32)
        model = self._make_model_proto([w1, w2])
        result = widget_small.get_weights_from_onnx(model)
        assert len(result) == 2
        np.testing.assert_array_almost_equal(result[0], w1)
        np.testing.assert_array_almost_equal(result[1], w2)

    def test_skips_non_gemm_nodes(self, widget_small):
        import onnx
        model = onnx.helper.make_model(
            onnx.helper.make_graph([], "g", [], [])
        )
        relu = onnx.helper.make_node("Relu", inputs=["x"], outputs=["y"])
        model.graph.node.append(relu)
        result = widget_small.get_weights_from_onnx(model)
        assert result == []

    def test_matmul_nodes_are_extracted(self, widget_small):
        import onnx
        from onnx import numpy_helper
        arr = np.ones((3, 4), dtype=np.float32)
        model = onnx.helper.make_model(
            onnx.helper.make_graph([], "g", [], [])
        )
        tensor = numpy_helper.from_array(arr, name="w0")
        model.graph.initializer.append(tensor)
        node = onnx.helper.make_node("MatMul", inputs=["in0", "w0"], outputs=["out0"])
        model.graph.node.append(node)
        result = widget_small.get_weights_from_onnx(model)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# wheelEvent
# ---------------------------------------------------------------------------

class TestWheelEvent:
    def test_zoom_in_increases_scale(self, widget_small):
        widget_small.show()
        widget_small.resize(400, 400)
        initial_scale = widget_small.transform().m11()
        # Simulate scroll up (zoom in)
        QTest.mouseMove(widget_small)
        event = QWheelEvent(
            QPoint(200, 200),
            widget_small.mapToGlobal(QPoint(200, 200)),
            QPoint(0, 120),
            QPoint(0, 120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        widget_small.wheelEvent(event)
        new_scale = widget_small.transform().m11()
        # Should zoom in (scale increases) or stay clamped
        assert new_scale >= initial_scale

    def test_wheel_event_is_accepted(self, widget_small):
        event = QWheelEvent(
            QPoint(200, 200),
            widget_small.mapToGlobal(QPoint(200, 200)),
            QPoint(0, 120),
            QPoint(0, 120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        widget_small.wheelEvent(event)
        assert event.isAccepted()


# ---------------------------------------------------------------------------
# keyPressEvent
# ---------------------------------------------------------------------------

class TestKeyPressEvent:
    def test_r_key_calls_go_to_node(self, widget_small):
        widget_small.go_to_node = Mock()
        QTest.keyClick(widget_small, Qt.Key.Key_R)
        widget_small.go_to_node.assert_called_once()


# ---------------------------------------------------------------------------
# hideEvent / showEvent
# ---------------------------------------------------------------------------

class TestHideShowEvents:
    def test_hide_calls_remove_settings(self, widget_small):
        remove_fn = Mock()
        widget_small.show()
        widget_small.remove_settings = [remove_fn]
        widget_small.hide()
        remove_fn.assert_called_once()
        assert widget_small.remove_settings == []

    def test_hide_clears_remove_settings_list(self, widget_small):
        widget_small.show()
        widget_small.remove_settings = [Mock(), Mock()]
        widget_small.hide()
        assert widget_small.remove_settings == []

    @patch("nn_verification_visualisation.view.network_view.network_widget.SettingsDialog.add_setting")
    def test_show_adds_settings_when_not_selectable(self, mock_add, widget_small):
        mock_add.return_value = Mock()
        widget_small.show()
        assert mock_add.call_count == 2

    @patch("nn_verification_visualisation.view.network_view.network_widget.SettingsDialog.add_setting")
    def test_show_does_not_add_settings_when_selectable(self, mock_add, qapp, small_config, qtbot):
        w = NetworkWidget(small_config, nodes_selectable=True)
        qtbot.addWidget(w)
        mock_add.reset_mock()
        w.show()
        mock_add.assert_not_called()


# ---------------------------------------------------------------------------
# go_to_node
# ---------------------------------------------------------------------------

class TestGoToNode:
    def test_go_to_node_starts_animation(self, widget_small):
        widget_small.go_to_node(0, 0)
        assert hasattr(widget_small, "anim_group")
        assert widget_small.anim_group is not None

    def test_go_to_node_empty_layers_does_nothing(self, widget_small):
        widget_small.node_layers = []
        # Should not raise
        widget_small.go_to_node(0, 0)