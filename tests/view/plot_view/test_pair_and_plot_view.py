from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from PySide6.QtCore import QSize
from PySide6.QtWidgets import QSpinBox, QWidget

from nn_verification_visualisation.view.plot_view.pair_loading_widget import PairLoadingWidget
from nn_verification_visualisation.view.plot_view.plot_view import PlotView
from nn_verification_visualisation.view.plot_view.status import Status


def _mock_insert_view_dependencies():
    tabs = QWidget()
    tabs.close_tab = Mock()
    tabs.setCornerWidget = Mock()
    tabs.reset = Mock()
    tabs.add_tab = Mock()
    tabs.widget = Mock()
    tabs.tabBar = Mock(return_value=Mock(height=Mock(return_value=40), sizeHint=Mock(return_value=QSize(100, 40))))

    action_menu = Mock()
    action_menu.menu = Mock()
    return tabs, action_menu


def test_pair_loading_widget_updates_ui_for_each_status(qapp):
    widget = PairLoadingWidget("Pair A", on_click=Mock())
    widget.show()

    widget.set_status(Status.Ongoing)
    assert widget._PairLoadingWidget__button.isVisible() is True
    assert widget._PairLoadingWidget__button.text() == "Cancel Execution"
    assert widget._PairLoadingWidget__title.text() == "Pair A - Loading"

    widget.set_status(Status.Done)
    assert widget._PairLoadingWidget__button.isVisible() is False
    assert widget._PairLoadingWidget__title.text() == "Pair A - Completed"

    widget.set_status(Status.Failed)
    assert widget._PairLoadingWidget__button.isVisible() is True
    assert widget._PairLoadingWidget__button.text() == "Show Error"
    assert widget._PairLoadingWidget__title.text() == "Pair A - Error"


def test_pair_loading_widget_button_click_delegates_to_callback(qapp):
    callback = Mock()
    widget = PairLoadingWidget("Pair A", on_click=callback)
    widget.set_status(Status.Ongoing)

    widget._PairLoadingWidget__button.click()

    callback.assert_called_once()


def test_plot_view_initializes_from_storage_and_adds_loading_tab(qapp):
    storage = MagicMock()
    storage.diagrams = [SimpleNamespace(name="D1")]
    tabs, action_menu = _mock_insert_view_dependencies()

    with (
        patch("nn_verification_visualisation.view.plot_view.plot_view.PlotViewController"),
        patch("nn_verification_visualisation.view.plot_view.plot_view.Storage", return_value=storage),
        patch("nn_verification_visualisation.view.base_view.insert_view.Tabs", return_value=tabs),
        patch("nn_verification_visualisation.view.base_view.insert_view.ActionMenu", return_value=action_menu),
        patch.object(PlotView, "add_plot_tab") as add_plot_tab,
    ):
        view = PlotView(Mock())

        add_plot_tab.assert_called_once_with(storage.diagrams[0])
        loading_widget = MagicMock()
        view.add_loading_tab(loading_widget)
        tabs.add_tab.assert_called_once_with(loading_widget)


def test_plot_view_close_tab_removes_diagram_from_storage(qapp):
    diagram = SimpleNamespace(name="D1")
    storage = MagicMock()
    storage.diagrams = [diagram]
    tabs, action_menu = _mock_insert_view_dependencies()

    with (
        patch("nn_verification_visualisation.view.plot_view.plot_view.PlotViewController"),
        patch("nn_verification_visualisation.view.plot_view.plot_view.Storage", return_value=storage),
        patch("nn_verification_visualisation.view.base_view.insert_view.Tabs", return_value=tabs),
        patch("nn_verification_visualisation.view.base_view.insert_view.ActionMenu", return_value=action_menu),
        patch.object(PlotView, "add_plot_tab"),
        patch("nn_verification_visualisation.view.plot_view.plot_view.InsertView.close_tab") as super_close_tab,
    ):
        view = PlotView(Mock())
        tabs.widget.return_value = SimpleNamespace(diagram_config=diagram)

        view.close_tab(0)

    assert storage.diagrams == []
    storage.request_autosave.assert_called_once()
    super_close_tab.assert_called_once_with(0)


def test_plot_view_show_and_hide_manage_settings_registration(qapp):
    storage = MagicMock()
    storage.diagrams = []
    remover = Mock()
    tabs, action_menu = _mock_insert_view_dependencies()

    with (
        patch("nn_verification_visualisation.view.plot_view.plot_view.PlotViewController"),
        patch("nn_verification_visualisation.view.plot_view.plot_view.Storage", return_value=storage),
        patch("nn_verification_visualisation.view.base_view.insert_view.Tabs", return_value=tabs),
        patch("nn_verification_visualisation.view.base_view.insert_view.ActionMenu", return_value=action_menu),
        patch("nn_verification_visualisation.view.plot_view.plot_view.SettingsDialog.add_setting", return_value=remover),
    ):
        view = PlotView(Mock())
        view.showEvent(None)
        assert view.settings_remover is remover

        view.hideEvent(None)

    remover.assert_called_once()
    assert view.settings_remover is None


def test_plot_view_direction_changer_updates_storage(qapp):
    storage = MagicMock()
    storage.diagrams = []
    storage.num_directions = 32
    tabs, action_menu = _mock_insert_view_dependencies()

    with (
        patch("nn_verification_visualisation.view.plot_view.plot_view.PlotViewController"),
        patch("nn_verification_visualisation.view.plot_view.plot_view.Storage", return_value=storage),
        patch("nn_verification_visualisation.view.base_view.insert_view.Tabs", return_value=tabs),
        patch("nn_verification_visualisation.view.base_view.insert_view.ActionMenu", return_value=action_menu),
    ):
        view = PlotView(Mock())
        changer = view.get_num_directions_changer()

        assert isinstance(changer, QSpinBox)
        changer.setValue(64)

    assert storage.num_directions == 64
