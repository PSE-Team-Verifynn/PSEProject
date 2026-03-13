import pytest
from PySide6.QtWidgets import QWidget, QSplitter, QScrollArea

from nn_verification_visualisation.view.base_view.tab import Tab

@pytest.fixture(autouse=True)
def patch_qtimer(mocker):
    """Patch QTimer.singleShot globally to prevent deferred splitter callbacks
    from firing after widgets are destroyed during other tests."""
    mocker.patch("nn_verification_visualisation.view.base_view.tab.QTimer.singleShot")

class ConcreteTab(Tab):
    """Minimal concrete subclass for testing."""

    def get_content(self) -> QWidget:
        w = QWidget()
        w.setObjectName("content")
        return w

    def get_side_bar(self) -> QWidget:
        w = QWidget()
        w.setObjectName("sidebar")
        return w


class TestTabInit:
    def test_title_is_stored(self, qtbot):
        tab = ConcreteTab("My Tab")
        qtbot.addWidget(tab)
        assert tab.title == "My Tab"

    def test_icon_defaults_to_none(self, qtbot):
        tab = ConcreteTab("Tab")
        qtbot.addWidget(tab)
        assert tab.icon is None

    def test_icon_is_stored(self, qtbot):
        tab = ConcreteTab("Tab", icon=":assets/icons/test.svg")
        qtbot.addWidget(tab)
        assert tab.icon == ":assets/icons/test.svg"

    def test_remove_close_button_defaults_to_false(self, qtbot):
        tab = ConcreteTab("Tab")
        qtbot.addWidget(tab)
        assert tab.remove_close_button is False

    def test_remove_close_button_can_be_set(self, qtbot):
        tab = ConcreteTab("Tab", remove_close_button=True)
        qtbot.addWidget(tab)
        assert tab.remove_close_button is True


class TestTabLayout:
    def test_with_sidebar_contains_splitter(self, qtbot):
        tab = ConcreteTab("Tab", has_sidebar=True)
        qtbot.addWidget(tab)
        splitters = tab.findChildren(QSplitter)
        assert len(splitters) == 1

    def test_with_sidebar_splitter_has_two_widgets(self, qtbot):
        tab = ConcreteTab("Tab", has_sidebar=True)
        qtbot.addWidget(tab)
        splitter = tab.findChildren(QSplitter)[0]
        assert splitter.count() == 2

    def test_without_sidebar_has_no_splitter(self, qtbot):
        tab = ConcreteTab("Tab", has_sidebar=False)
        qtbot.addWidget(tab)
        splitters = tab.findChildren(QSplitter)
        assert len(splitters) == 0

    def test_content_container_object_name(self, qtbot):
        tab = ConcreteTab("Tab", has_sidebar=False)
        qtbot.addWidget(tab)
        containers = [w for w in tab.findChildren(QWidget) if w.objectName() == "tab-content"]
        assert len(containers) == 1

    def test_sidebar_container_object_name(self, qtbot):
        tab = ConcreteTab("Tab", has_sidebar=True)
        qtbot.addWidget(tab)
        sidebars = [w for w in tab.findChildren(QWidget) if w.objectName() == "tab-sidebar"]
        assert len(sidebars) == 1