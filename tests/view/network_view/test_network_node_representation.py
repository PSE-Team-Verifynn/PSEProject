from unittest.mock import Mock

from PySide6.QtCore import QRectF
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView, QStyleOptionGraphicsItem

from nn_verification_visualisation.view.network_view.network_node_representation import NetworkLayerLine, NetworkNode


def test_network_layer_line_sets_expected_geometry_and_pen(qapp):
    line = NetworkLayerLine(10.0, 5.0, 25.0)

    assert line.line().x1() == 10.0
    assert line.line().y1() == 5.0
    assert line.line().y2() == 25.0
    assert line.pen().width() == 3
    assert line.zValue() == 0


def test_network_node_mouse_press_toggles_selection_and_calls_callback(qapp):
    callback = Mock()
    node = NetworkNode(2, 1, 8.0, callback, selectable=True)
    event = Mock()

    node.mousePressEvent(event)

    assert node.selectable is True
    callback.assert_called_once_with((1, 2))
    event.accept.assert_called_once()


def test_network_node_ignores_event_when_too_small_in_lod_mode(qapp):
    callback = Mock()
    node = NetworkNode(0, 0, 8.0, callback, selectable=True)
    scene = QGraphicsScene()
    view = QGraphicsView(scene)
    scene.addItem(node)
    view.scale(0.01, 0.01)
    node.lod_threshold = 0.5
    node.set_lod_mode(True)
    event = Mock()

    node.mousePressEvent(event)

    callback.assert_not_called()
    event.ignore.assert_called_once()


def test_network_node_paint_skips_when_lod_below_threshold(qapp):
    node = NetworkNode(0, 0, 8.0, None, selectable=False)
    node.set_lod_mode(True)
    node.lod_threshold = 2.0
    image = QImage(40, 40, QImage.Format.Format_ARGB32)
    image.fill(0)
    painter = QPainter(image)
    option = QStyleOptionGraphicsItem()
    option.exposedRect = QRectF(0, 0, 40, 40)

    node.paint(painter, option)
    painter.end()

    assert image.pixelColor(20, 20).alpha() == 0
