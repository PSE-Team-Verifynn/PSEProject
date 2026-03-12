import pytest
from unittest.mock import Mock
from PySide6.QtCore import QPointF

from nn_verification_visualisation.view.network_view.network_edge_representation import NetworkEdgeBatch


def make_node(x, y):
    node = Mock()
    node.scenePos.return_value = QPointF(x, y)
    return node


class TestNetworkEdgeBatchBlockMode:

    def test_block_has_minimum_height_when_single_node_per_layer(self, qapp):
        """Regression: a single-node layer must not produce a zero-height block."""
        source = [make_node(0, 100)]
        target = [make_node(200, 100)]

        batch = NetworkEdgeBatch(source, target, force_block=True)

        assert batch.boundingRect().height() >= batch._NetworkEdgeBatch__minimum_block_height

    def test_block_has_minimum_height_when_nodes_are_at_identical_positions(self, qapp):
        """Regression: nodes at the exact same y must still produce a visible block."""
        source = [make_node(0, 50), make_node(0, 50)]  # collapsed
        target = [make_node(200, 50), make_node(200, 50)]

        batch = NetworkEdgeBatch(source, target, force_block=True)

        assert batch.boundingRect().height() >= batch._NetworkEdgeBatch__minimum_block_height