import pytest
from PySide6.QtWidgets import QWidget

from nn_verification_visualisation.view.base_view.base_view import BaseView
from nn_verification_visualisation.view.base_view.color_manager import ColorManager
from tests.conftest import mock_color_manager


@pytest.fixture
def mocked_base_view(mocker, qtbot, mock_color_manager):
    """Set up common mocks for BaseView tests."""
    mock_network_view = QWidget()
    mock_plot_view = QWidget()

    mocks = {'network_view_class': mocker.patch('nn_verification_visualisation.view.base_view.base_view.NetworkView',
                                                return_value=mock_network_view),
             'plot_view_class': mocker.patch('nn_verification_visualisation.view.base_view.base_view.PlotView',
                                             return_value=mock_plot_view),
             'network_view': mock_network_view,
             'plot_view': mock_plot_view,
             'color_manager': mock_color_manager,
             }

    mocks['base_view'] = BaseView(mocks['color_manager'])

    qtbot.addWidget(mocks['base_view'])

    return mocks


class TestBaseView:
    def test_views_initialized(self, mocked_base_view):
        mocked_base_view['network_view_class'].assert_called_once_with(mocked_base_view['base_view'].change_active_view,
                                                                       parent=mocked_base_view['base_view'])
        mocked_base_view['plot_view_class'].assert_called_once_with(mocked_base_view['base_view'].change_active_view,
                                                                    parent=mocked_base_view['base_view'])

    def test_adds_view_to_the_stack(self, mocked_base_view):
        layout = mocked_base_view['base_view'].stack.layout()
        widgets = [layout.widget(i) for i in range(layout.count())]
        assert mocked_base_view['network_view'] in widgets
        assert mocked_base_view['plot_view'] in widgets

    def test_initially_sets_up_network_view_and_colors(self, mocked_base_view):
        assert mocked_base_view['base_view'].active_view == mocked_base_view['network_view']
        assert mocked_base_view['base_view'].stack.currentIndex() == 0

    def test_change_active_view_sets_diagram_view_and_colors(self, mocked_base_view):
        mocked_base_view['base_view'].change_active_view()

        assert mocked_base_view['base_view'].active_view == mocked_base_view['plot_view']
        mocked_base_view['color_manager'].set_colors.assert_called_with(ColorManager.DIAGRAM_COLORS)
        assert mocked_base_view['base_view'].stack.currentIndex() == 1

    def test_change_active_view_twice_switches_back(self, mocked_base_view):
        mocked_base_view['base_view'].change_active_view()
        mocked_base_view['base_view'].change_active_view()

        assert mocked_base_view['base_view'].active_view == mocked_base_view['network_view']
        mocked_base_view['color_manager'].set_colors.assert_called_with(ColorManager.NETWORK_COLORS)
        assert mocked_base_view['base_view'].stack.currentIndex() == 0
