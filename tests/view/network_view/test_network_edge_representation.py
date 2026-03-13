import pytest
import numpy as np
from unittest.mock import Mock, call
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QStyleOptionGraphicsItem

from nn_verification_visualisation.view.network_view.network_edge_representation import NetworkEdgeBatch


def make_node(x, y):
    node = Mock()
    node.scenePos.return_value = QPointF(x, y)
    return node


def make_painter():
    return Mock(spec=QPainter)

@pytest.fixture
def default_nodes():
    source = [make_node(0, 0), make_node(0, 100)]
    target = [make_node(200, 0), make_node(200, 100)]
    return source, target


class TestNetworkEdgeBatchBlockModeGeometry:
    def test_empty_source_produces_empty_bounding_rect(self, qapp):
        batch = NetworkEdgeBatch([], [make_node(200, 100)], force_block=True)
        assert batch.boundingRect() == QRectF()

    def test_empty_target_produces_empty_bounding_rect(self, qapp):
        batch = NetworkEdgeBatch([make_node(0, 100)], [], force_block=True)
        assert batch.boundingRect() == QRectF()

    def test_both_empty_produces_empty_bounding_rect(self, qapp):
        batch = NetworkEdgeBatch([], [], force_block=True)
        assert batch.boundingRect() == QRectF()

    def test_polygon_x_spans_source_to_target(self, qapp):
        source = [make_node(0, 50), make_node(0, 150)]
        target = [make_node(200, 60), make_node(200, 140)]
        batch = NetworkEdgeBatch(source, target, force_block=True)

        rect = batch.boundingRect()
        assert rect.left() == pytest.approx(0.0)
        assert rect.right() == pytest.approx(200.0)

    def test_polygon_y_spans_first_to_last_source_node(self, qapp):
        source = [make_node(0, 50), make_node(0, 150)]
        target = [make_node(200, 60), make_node(200, 140)]
        batch = NetworkEdgeBatch(source, target, force_block=True)

        rect = batch.boundingRect()
        assert rect.top() == pytest.approx(50.0)
        assert rect.bottom() == pytest.approx(150.0)

    def test_single_node_per_layer_does_not_crash(self, qapp):
        """Regression: single-node layers should not raise."""
        source = [make_node(0, 100)]
        target = [make_node(200, 100)]
        batch = NetworkEdgeBatch(source, target, force_block=True)
        assert batch.boundingRect().width() == pytest.approx(200.0)

    def test_block_mode_sets_z_value_minus_one(self, qapp, default_nodes):
        source, target = default_nodes
        batch = NetworkEdgeBatch(source, target, force_block=True)
        assert batch.zValue() == -1

class TestNetworkEdgeBatchLineModeGeometry:
    def test_line_count_equals_source_times_target(self, qapp):
        source = [make_node(0, y) for y in [0, 50, 100]]
        target = [make_node(200, y) for y in [0, 75]]
        batch = NetworkEdgeBatch(source, target)
        assert len(batch.lines_data) == 6  # 3 × 2

    def test_line_weight_defaults_to_one_without_weights(self, qapp):
        source = [make_node(0, 0)]
        target = [make_node(200, 0)]
        batch = NetworkEdgeBatch(source, target)
        _, w = batch.lines_data[0]
        assert w == pytest.approx(1.0)

    def test_bounding_rect_x_starts_at_source(self, qapp):
        source = [make_node(10, 20), make_node(10, 80)]
        target = [make_node(300, 20), make_node(300, 80)]
        batch = NetworkEdgeBatch(source, target)
        assert batch.boundingRect().x() == pytest.approx(10.0)
        assert batch.boundingRect().width() == pytest.approx(290.0)

    def test_bounding_rect_y_uses_global_min_max(self, qapp):
        # min_y = min(source[0].y, target[0].y), max_y = max(source[-1].y, target[-1].y)
        source = [make_node(0, 30), make_node(0, 90)]
        target = [make_node(200, 10), make_node(200, 110)]
        batch = NetworkEdgeBatch(source, target)
        rect = batch.boundingRect()
        assert rect.y() == pytest.approx(10.0)
        assert rect.height() == pytest.approx(100.0)

    def test_line_mode_sets_z_value_minus_one(self, qapp):
        source = [make_node(0, 0)]
        target = [make_node(100, 0)]
        batch = NetworkEdgeBatch(source, target)
        assert batch.zValue() == -1

    def test_min_max_weight_defaults_when_no_weights(self, qapp):
        source = [make_node(0, 0)]
        target = [make_node(100, 0)]
        batch = NetworkEdgeBatch(source, target)
        assert batch.min_weight == pytest.approx(0.0)
        assert batch.max_weight == pytest.approx(1.0)


class TestNetworkEdgeBatchWeights:
    def test_standard_format_weights_stored_per_line(self, qapp, default_nodes):
        """If every weight lookup raises, min/max stay at 0.0/1.0."""
        source, target = default_nodes
        weights = np.array([[0.1, 0.2], [0.3, 0.4]])
        batch = NetworkEdgeBatch(source, target, use_weighted=True, weights=weights)

        # lines are created as: (s0,t0), (s0,t1), (s1,t0), (s1,t1)
        assert batch.lines_data[0][1] == pytest.approx(0.1)
        assert batch.lines_data[1][1] == pytest.approx(0.2)
        assert batch.lines_data[2][1] == pytest.approx(0.3)
        assert batch.lines_data[3][1] == pytest.approx(0.4)

    def test_weight_min_max_computed_from_abs_values(self, qapp):
        source = [make_node(0, 0), make_node(0, 100)]
        target = [make_node(200, 0)]
        weights = np.array([[3.0], [-7.0]])  # abs: 3, 7
        batch = NetworkEdgeBatch(source, target, use_weighted=True, weights=weights)
        assert batch.min_weight == pytest.approx(3.0)
        assert batch.max_weight == pytest.approx(7.0)

    def test_index_error_falls_back_to_zero_weight(self, qapp, default_nodes):
        source, target = default_nodes
        # shape (1,1) is too small for 2×2 node grid
        weights = np.array([[1.0]])
        batch = NetworkEdgeBatch(source, target, use_weighted=True, weights=weights)
        # line(1,1) is out of bounds → w=0.0
        assert batch.lines_data[3][1] == pytest.approx(0.0)

    def test_all_failed_lookups_keep_fallback_range(self, qapp, default_nodes):
        """If every weight lookup raises, min/max stay at 0.0/1.0."""
        source, target = default_nodes
        weights = np.array([])  # empty → IndexError on any access
        batch = NetworkEdgeBatch(source, target, use_weighted=True, weights=weights)
        assert batch.min_weight == pytest.approx(0.0)
        assert batch.max_weight == pytest.approx(1.0)


class TestNetworkEdgeBatchPaintBlock:
    def test_paint_block_calls_draw_polygon(self, qapp, default_nodes):
        source, target = default_nodes
        batch = NetworkEdgeBatch(source, target, force_block=True)

        painter = make_painter()
        batch.paint(painter, QStyleOptionGraphicsItem())

        painter.drawPolygon.assert_called_once()

    def test_paint_block_sets_no_pen(self, qapp, default_nodes):
        from PySide6.QtCore import Qt
        source, target = default_nodes
        batch = NetworkEdgeBatch(source, target, force_block=True)

        painter = make_painter()
        batch.paint(painter, QStyleOptionGraphicsItem())

        assert call(Qt.NoPen) in painter.setPen.call_args_list

    def test_paint_block_sets_brush(self, qapp, default_nodes):
        source, target = default_nodes
        batch = NetworkEdgeBatch(source, target, force_block=True)

        painter = make_painter()
        batch.paint(painter, QStyleOptionGraphicsItem())

        painter.setBrush.assert_called_once()

    def test_paint_block_does_not_call_draw_lines(self, qapp, default_nodes):
        source, target = default_nodes
        batch = NetworkEdgeBatch(source, target, force_block=True)

        painter = make_painter()
        batch.paint(painter, QStyleOptionGraphicsItem())

        painter.drawLines.assert_not_called()
        painter.drawLine.assert_not_called()


# ---------------------------------------------------------------------------
# Paint – line mode (normal / unweighted)
# ---------------------------------------------------------------------------

class TestNetworkEdgeBatchPaintLineNormal:
    def test_paint_normal_calls_draw_lines_once(self, qapp):
        source = [make_node(0, y) for y in [0, 50, 100]]
        target = [make_node(200, y) for y in [0, 50, 100]]
        batch = NetworkEdgeBatch(source, target)

        painter = make_painter()
        batch.paint(painter, QStyleOptionGraphicsItem())

        painter.drawLines.assert_called_once()

    def test_paint_normal_passes_all_lines(self, qapp):
        source = [make_node(0, y) for y in [0, 50, 100]]
        target = [make_node(200, y) for y in [0, 50, 100]]
        batch = NetworkEdgeBatch(source, target)

        painter = make_painter()
        batch.paint(painter, QStyleOptionGraphicsItem())

        drawn = painter.drawLines.call_args[0][0]
        assert len(drawn) == 9  # 3 × 3

    def test_paint_weighted_mode_without_weights_uses_normal_path(self, qapp):
        """use_weighted=True but weights=None → normal draw path (drawLines)."""
        source = [make_node(0, 0)]
        target = [make_node(200, 0)]
        batch = NetworkEdgeBatch(source, target, use_weighted=True, weights=None)

        painter = make_painter()
        batch.paint(painter, QStyleOptionGraphicsItem())

        painter.drawLines.assert_called_once()
        painter.drawLine.assert_not_called()


# ---------------------------------------------------------------------------
# Paint – line mode (weighted)
# ---------------------------------------------------------------------------

class TestNetworkEdgeBatchPaintLineWeighted:
    def test_paint_weighted_calls_draw_line_per_edge(self, qapp):
        source = [make_node(0, y) for y in [0, 100]]
        target = [make_node(200, y) for y in [0, 100]]
        weights = np.array([[0.5, 0.8], [0.2, 1.0]])
        batch = NetworkEdgeBatch(source, target, use_weighted=True, weights=weights)

        painter = make_painter()
        batch.paint(painter, QStyleOptionGraphicsItem())

        assert painter.drawLine.call_count == 4
        painter.drawLines.assert_not_called()

    def test_paint_weighted_uniform_weight_takes_weighted_path(self, qapp):
        """min==max, non-zero → normalized=1.0; still uses per-line draw path."""
        source = [make_node(0, 0)]
        target = [make_node(200, 0)]
        weights = np.array([[2.0]])
        batch = NetworkEdgeBatch(source, target, use_weighted=True, weights=weights)

        painter = make_painter()
        batch.paint(painter, QStyleOptionGraphicsItem())

        painter.drawLine.assert_called_once()
        painter.drawLines.assert_not_called()

    def test_paint_weighted_zero_weight_draws_all_lines(self, qapp):
        """Zero weights are still drawn (with min alpha)."""
        source = [make_node(0, 0), make_node(0, 100)]
        target = [make_node(200, 0)]
        weights = np.array([[0.0], [0.0]])
        batch = NetworkEdgeBatch(source, target, use_weighted=True, weights=weights)

        painter = make_painter()
        batch.paint(painter, QStyleOptionGraphicsItem())

        assert painter.drawLine.call_count == 2