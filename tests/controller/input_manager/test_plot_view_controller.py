from unittest.mock import MagicMock

from PySide6.QtWidgets import QApplication
from nn_verification_visualisation.controller.input_manager.plot_view_controller import PlotViewController
from nn_verification_visualisation.view.base_view.color_manager import ColorManager
from nn_verification_visualisation.view.plot_view.plot_view import PlotView


def test_compute_polygon():
    config = MagicMock()
    test_polygon = PlotViewController.compute_polygon(PlotViewController(current_plot_view=config), [(0, 2),(-1,1)], [(0, 1), (1, 0)])
    assert test_polygon == [(1.0, 2.0),(-1.0,2.0),(-1.0,0.0) ,(1.0,0.0)]