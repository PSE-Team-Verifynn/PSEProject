import pytest
from unittest.mock import Mock
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import QSize

from nn_verification_visualisation.view.base_view.tabs import Tabs, PersistentTabBar
from nn_verification_visualisation.view.base_view.tab import Tab


class ConcreteTab(Tab):
    """Minimal concrete Tab for testing."""

    def __init__(self, title="Test Tab", icon=None, remove_close_button=False):
        super().__init__(title, icon=icon, remove_close_button=remove_close_button)

    def get_content(self):
        return QWidget()

    def get_side_bar(self):
        return QWidget()

@pytest.fixture(autouse=True)
def patch_qtimer(mocker):
    """Patch QTimer.singleShot globally to prevent deferred splitter callbacks
    from firing after widgets are destroyed during other tests."""
    mocker.patch("nn_verification_visualisation.view.base_view.tab.QTimer.singleShot")

@pytest.fixture
def empty_page():
    w = QWidget()
    return w


@pytest.fixture
def tabs_with_empty(qtbot, empty_page):
    t = Tabs(empty_page=empty_page)
    qtbot.addWidget(t)
    return t


@pytest.fixture
def tabs_plain(qtbot):
    t = Tabs()
    qtbot.addWidget(t)
    return t


class TestTabsInit:
    def test_has_empty_page_when_provided(self, tabs_with_empty, empty_page):
        assert tabs_with_empty.has_empty_page is True
        assert tabs_with_empty.empty_page is empty_page

    def test_initial_tab_count_with_empty_page(self, tabs_with_empty):
        assert tabs_with_empty.count() == 1

    def test_initial_tab_count_without_empty_page(self, tabs_plain):
        assert tabs_plain.count() == 0

    def test_welcome_tab_has_no_close_button(self, tabs_with_empty):
        bar = tabs_with_empty.tabBar()
        assert bar.tabButton(0, bar.ButtonPosition.RightSide) is None

    def test_uses_persistent_tab_bar(self, tabs_plain):
        assert isinstance(tabs_plain.tabBar(), PersistentTabBar)

    def test_custom_on_close_callback_is_connected(self, qtbot):
        on_close = Mock()
        t = Tabs(on_close=on_close)
        qtbot.addWidget(t)
        tab = ConcreteTab()
        t.add_tab(tab)
        t.tabCloseRequested.emit(0)
        on_close.assert_called_once_with(0)


class TestAddTab:
    def test_add_first_tab_does_not_increase_count(self, tabs_with_empty):
        tabs_with_empty.add_tab(ConcreteTab())
        assert tabs_with_empty.count() == 1  # empty page replaced

    def test_add_tab_replaces_empty_page(self, tabs_with_empty, empty_page):
        tabs_with_empty.add_tab(ConcreteTab())
        widgets = [tabs_with_empty.widget(i) for i in range(tabs_with_empty.count())]
        assert empty_page not in widgets

    def test_add_tab_focuses_new_tab_by_default(self, tabs_with_empty):
        tab = ConcreteTab()
        tabs_with_empty.add_tab(tab)
        assert tabs_with_empty.currentWidget() is tab

    def test_add_silent_does_not_focus_new_tab(self, tabs_with_empty):
        first = ConcreteTab("First")
        tabs_with_empty.add_tab(first)
        second = ConcreteTab("Second")
        tabs_with_empty.add_tab(second, add_silent=True)
        assert tabs_with_empty.currentWidget() is first

    def test_add_tab_at_index(self, tabs_with_empty):
        tab1 = ConcreteTab("First")
        tab2 = ConcreteTab("Second")
        tabs_with_empty.add_tab(tab1)
        tabs_with_empty.add_tab(tab2, index=0)
        assert tabs_with_empty.widget(0) is tab2

    def test_add_multiple_tabs(self, tabs_plain):
        for i in range(3):
            tabs_plain.add_tab(ConcreteTab(f"Tab {i}"))
        assert tabs_plain.count() == 3

    def test_remove_close_button_when_flagged(self, tabs_plain):
        tab = ConcreteTab(remove_close_button=True)
        tabs_plain.add_tab(tab)
        bar = tabs_plain.tabBar()
        idx = tabs_plain.indexOf(tab)
        assert bar.tabButton(idx, bar.ButtonPosition.RightSide) is None

    def test_add_tab_without_empty_page(self, tabs_plain):
        tab = ConcreteTab()
        tabs_plain.add_tab(tab)
        assert tabs_plain.count() == 1
        assert tabs_plain.currentWidget() is tab


class TestCloseTab:
    def test_close_tab_decreases_count(self, tabs_plain):
        tabs_plain.add_tab(ConcreteTab())
        tabs_plain.close_tab(0)
        assert tabs_plain.count() == 0

    def test_close_last_tab_restores_empty_page(self, tabs_with_empty, empty_page):
        tabs_with_empty.add_tab(ConcreteTab())
        tabs_with_empty.close_tab(0)
        assert tabs_with_empty.count() == 1
        assert tabs_with_empty.currentWidget() is empty_page

    def test_close_one_of_many_tabs(self, tabs_plain):
        tab1 = ConcreteTab("First")
        tab2 = ConcreteTab("Second")
        tabs_plain.add_tab(tab1)
        tabs_plain.add_tab(tab2)
        tabs_plain.close_tab(0)
        assert tabs_plain.count() == 1
        assert tabs_plain.widget(0) is tab2

    def test_close_tab_no_empty_page_no_restore(self, tabs_plain):
        tabs_plain.add_tab(ConcreteTab())
        tabs_plain.close_tab(0)
        assert tabs_plain.count() == 0


class TestReset:
    def test_reset_removes_all_tabs(self, tabs_plain):
        for _ in range(3):
            tabs_plain.add_tab(ConcreteTab())
        tabs_plain.reset()
        assert tabs_plain.count() == 0

    def test_reset_restores_empty_page(self, tabs_with_empty, empty_page):
        tabs_with_empty.add_tab(ConcreteTab())
        tabs_with_empty.add_tab(ConcreteTab())
        tabs_with_empty.reset()
        assert tabs_with_empty.count() == 1
        assert tabs_with_empty.currentWidget() is empty_page


class TestUpdateEmptyState:
    def test_empty_page_shown_when_no_real_tabs(self, tabs_with_empty, empty_page):
        assert tabs_with_empty.currentWidget() is empty_page

    def test_empty_page_not_current_when_real_tab_added(self, tabs_with_empty, empty_page):
        tabs_with_empty.add_tab(ConcreteTab())
        assert tabs_with_empty.currentWidget() is not empty_page

    def test_no_op_when_no_empty_page(self, tabs_plain):
        # _update_empty_state should be safe to call without empty_page
        tabs_plain._update_empty_state()  # should not raise


class TestPersistentTabBar:
    def test_size_hint_with_no_tabs(self, qtbot):
        bar = PersistentTabBar()
        qtbot.addWidget(bar)
        hint = bar.sizeHint()
        assert hint == QSize(0, 48)

    def test_size_hint_with_tabs_is_not_fixed(self, qtbot):
        tabs = Tabs()
        qtbot.addWidget(tabs)
        bar = tabs.tabBar()
        tabs.add_tab(ConcreteTab())
        hint = bar.sizeHint()
        assert hint != QSize(0, 48)

    def test_draw_base_disabled(self, qtbot):
        bar = PersistentTabBar()
        qtbot.addWidget(bar)
        assert not bar.drawBase()