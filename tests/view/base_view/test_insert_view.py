import pytest
from PySide6.QtWidgets import QWidget, QPushButton
from PySide6.QtCore import Qt, QSize
from unittest.mock import Mock

from nn_verification_visualisation.view.base_view.insert_view import InsertView


class MockDialog(QWidget):
    """Simple mock dialog for testing."""

    def __init__(self, close_callback, title="Test"):
        super().__init__()
        self.close_callback = close_callback
        self.title = title


@pytest.fixture
def mocked_insert_view(mocker, qtbot):
    """Set up common mocks for InsertView tests."""
    # Create real QWidget for Tabs since it needs to be added to layouts
    mock_tabs = QWidget()
    mock_tabs.close_tab = Mock()
    mock_tabs.setCornerWidget = Mock()
    mock_tab_bar = Mock()
    mock_tab_bar.height.return_value = 40
    mock_tab_bar.sizeHint.return_value = QSize(100, 40)
    mock_tabs.tabBar = Mock(return_value=mock_tab_bar)

    mock_action_menu = Mock()
    mock_action_menu.menu = Mock()
    mock_action_menu.hide = Mock()

    # Patch before creating InsertView
    tabs_class = mocker.patch('nn_verification_visualisation.view.base_view.insert_view.Tabs', return_value=mock_tabs)
    action_menu_class = mocker.patch('nn_verification_visualisation.view.base_view.insert_view.ActionMenu',
                                     return_value=mock_action_menu)

    mocks = {
        'tabs_class': tabs_class,
        'action_menu_class': action_menu_class,
        'tabs': mock_tabs,
        'action_menu': mock_action_menu,
    }

    mocks['insert_view'] = InsertView()
    qtbot.addWidget(mocks['insert_view'])

    # Show the widget and set a reasonable size for testing
    mocks['insert_view'].show()
    mocks['insert_view'].resize(640, 480)
    qtbot.waitExposed(mocks['insert_view'])

    return mocks


class TestInsertViewInitialization:
    """Tests for InsertView initialization."""

    def test_tabs_initialized(self, mocked_insert_view):
        """Tabs is created with close_tab callback and empty_page=get_default_tab()."""
        mocked_insert_view['tabs_class'].assert_called_once_with(
            mocked_insert_view['insert_view'].close_tab,
            empty_page=None  # get_default_tab() returns None by default
        )

    def test_default_tab_returns_none(self, mocked_insert_view):
        """get_default_tab() returns None by default."""
        assert mocked_insert_view['insert_view'].get_default_tab() is None

    def test_dialog_stack_initially_empty(self, mocked_insert_view):
        """Dialog stack is empty on initialization."""
        assert len(mocked_insert_view['insert_view']._InsertView__dialog_stack) == 0

    def test_action_menu_initially_none(self, mocked_insert_view):
        """action_menu is None on initialization."""
        assert mocked_insert_view['insert_view'].action_menu is None

    def test_action_menu_open_initially_false(self, mocked_insert_view):
        """action_menu_open is False on initialization."""
        assert mocked_insert_view['insert_view'].action_menu_open is False

    def test_menu_button_added_to_top_left_corner(self, mocked_insert_view):
        """Corner widget is set on TopLeftCorner."""
        mocked_insert_view['tabs'].setCornerWidget.assert_called_once()
        call_args = mocked_insert_view['tabs'].setCornerWidget.call_args
        assert call_args[0][1] == Qt.Corner.TopLeftCorner


class TestOpenDialog:
    """Tests for open_dialog method."""

    def test_open_dialog_adds_to_stack(self, mocked_insert_view, qtbot):
        """After calling open_dialog, the dialog appears in __dialog_stack."""
        dialog = MockDialog(mocked_insert_view['insert_view'].close_dialog, "Test Dialog")

        mocked_insert_view['insert_view'].open_dialog(dialog)

        assert len(mocked_insert_view['insert_view']._InsertView__dialog_stack) == 1
        assert mocked_insert_view['insert_view']._InsertView__dialog_stack[0] == dialog

    def test_open_dialog_sets_parent(self, mocked_insert_view, qtbot):
        """The dialog's parent is set to the view."""
        dialog = MockDialog(mocked_insert_view['insert_view'].close_dialog, "Test Dialog")

        mocked_insert_view['insert_view'].open_dialog(dialog)

        assert dialog.parent() == mocked_insert_view['insert_view']

    def test_open_dialog_sets_visible(self, mocked_insert_view, qtbot):
        """The dialog is made visible."""
        dialog = MockDialog(mocked_insert_view['insert_view'].close_dialog, "Test Dialog")

        mocked_insert_view['insert_view'].open_dialog(dialog)
        qtbot.waitExposed(dialog)

        assert dialog.isVisible() is True

    def test_open_dialog_sets_geometry(self, mocked_insert_view, qtbot):
        """The dialog geometry matches the view's rect."""
        dialog = MockDialog(mocked_insert_view['insert_view'].close_dialog, "Test Dialog")

        mocked_insert_view['insert_view'].open_dialog(dialog)

        assert dialog.geometry() == mocked_insert_view['insert_view'].rect()

    def test_open_multiple_dialogs_stacks_them(self, mocked_insert_view, qtbot):
        """Two dialogs can be open, both in the stack."""
        dialog1 = MockDialog(mocked_insert_view['insert_view'].close_dialog, "Dialog 1")
        dialog2 = MockDialog(mocked_insert_view['insert_view'].close_dialog, "Dialog 2")

        mocked_insert_view['insert_view'].open_dialog(dialog1)
        mocked_insert_view['insert_view'].open_dialog(dialog2)

        stack = mocked_insert_view['insert_view']._InsertView__dialog_stack
        assert len(stack) == 2
        assert stack[0] == dialog1
        assert stack[1] == dialog2


class TestCloseDialog:
    """Tests for close_dialog method."""

    def test_close_dialog_returns_true_on_success(self, mocked_insert_view, qtbot):
        """Returns True when dialog is closed."""
        dialog = MockDialog(mocked_insert_view['insert_view'].close_dialog, "Test Dialog")
        mocked_insert_view['insert_view'].open_dialog(dialog)

        result = mocked_insert_view['insert_view'].close_dialog()

        assert result is True

    def test_close_dialog_returns_false_when_stack_empty(self, mocked_insert_view):
        """Returns False with no open dialogs."""
        result = mocked_insert_view['insert_view'].close_dialog()

        assert result is False

    def test_close_dialog_removes_from_stack(self, mocked_insert_view, qtbot):
        """Stack is shorter after closing."""
        dialog = MockDialog(mocked_insert_view['insert_view'].close_dialog, "Test Dialog")
        mocked_insert_view['insert_view'].open_dialog(dialog)

        mocked_insert_view['insert_view'].close_dialog()

        assert len(mocked_insert_view['insert_view']._InsertView__dialog_stack) == 0

    def test_close_dialog_clears_parent(self, mocked_insert_view, qtbot):
        """setParent(None) is called on the removed dialog."""
        dialog = MockDialog(mocked_insert_view['insert_view'].close_dialog, "Test Dialog")
        mocked_insert_view['insert_view'].open_dialog(dialog)

        mocked_insert_view['insert_view'].close_dialog()

        assert dialog.parent() is None

    def test_close_dialog_pops_last_opened(self, mocked_insert_view, qtbot):
        """With two dialogs open, the most recently opened one is closed first (LIFO)."""
        dialog1 = MockDialog(mocked_insert_view['insert_view'].close_dialog, "Dialog 1")
        dialog2 = MockDialog(mocked_insert_view['insert_view'].close_dialog, "Dialog 2")

        mocked_insert_view['insert_view'].open_dialog(dialog1)
        mocked_insert_view['insert_view'].open_dialog(dialog2)

        # Close the first time - should remove dialog2
        mocked_insert_view['insert_view'].close_dialog()

        stack = mocked_insert_view['insert_view']._InsertView__dialog_stack
        assert len(stack) == 1
        assert stack[0] == dialog1
        assert dialog2.parent() is None
        assert dialog1.parent() == mocked_insert_view['insert_view']


class TestResizeEvent:
    """Tests for resizeEvent method."""

    def test_resize_updates_dialog_geometry(self, mocked_insert_view, qtbot):
        """When the widget is resized, open dialogs have their geometry updated to match."""
        dialog = MockDialog(mocked_insert_view['insert_view'].close_dialog, "Test Dialog")
        mocked_insert_view['insert_view'].open_dialog(dialog)

        # Resize the view
        mocked_insert_view['insert_view'].resize(500, 600)
        qtbot.wait(10)  # Give Qt time to process the resize event

        # Dialog geometry should match the new rect
        assert dialog.geometry() == mocked_insert_view['insert_view'].rect()

    def test_resize_with_no_dialogs_does_not_crash(self, mocked_insert_view, qtbot):
        """Resize with empty stack is safe."""
        # This should not raise any exceptions
        mocked_insert_view['insert_view'].resize(500, 600)
        qtbot.wait(10)
        assert len(mocked_insert_view['insert_view']._InsertView__dialog_stack) == 0


class TestCloseTab:
    """Tests for close_tab method."""

    def test_close_tab_delegates_to_tabs(self, mocked_insert_view):
        """close_tab(i) calls self.tabs.close_tab(i)."""
        mocked_insert_view['insert_view'].close_tab(3)

        mocked_insert_view['tabs'].close_tab.assert_called_once_with(3)


class TestSetBarCornerWidgets:
    """Tests for set_bar_corner_widgets method."""

    def test_set_bar_corner_widgets_adds_all_widgets(self, mocked_insert_view, qtbot):
        """All provided widgets appear in the container layout."""
        widget1 = QWidget()
        widget2 = QWidget()
        qtbot.addWidget(widget1)
        qtbot.addWidget(widget2)

        mocked_insert_view['insert_view'].set_bar_corner_widgets(
            [widget1, widget2],
            Qt.Corner.TopRightCorner
        )

        # Verify setCornerWidget was called with a container
        assert mocked_insert_view['tabs'].setCornerWidget.call_count == 2  # Once in __init__, once here
        last_call = mocked_insert_view['tabs'].setCornerWidget.call_args_list[-1]
        container = last_call[0][0]
        corner = last_call[0][1]

        assert corner == Qt.Corner.TopRightCorner

        # Check that widgets are children of the container
        layout = container.layout()
        layout_widgets = [layout.itemAt(i).widget() for i in range(layout.count()) if layout.itemAt(i).widget()]
        assert widget1 in layout_widgets
        assert widget2 in layout_widgets

    def test_set_bar_corner_widgets_sets_fixed_height(self, mocked_insert_view, qtbot):
        """Container height matches tab bar height."""
        widget = QWidget()
        qtbot.addWidget(widget)

        mocked_insert_view['insert_view'].set_bar_corner_widgets(
            [widget],
            Qt.Corner.TopRightCorner
        )

        last_call = mocked_insert_view['tabs'].setCornerWidget.call_args_list[-1]
        container = last_call[0][0]

        # The fixed height should match the tab bar's size hint height (from our mock: 40)
        expected_height = mocked_insert_view['tabs'].tabBar().sizeHint().height()
        assert container.height() == expected_height


class TestCreateSimpleIconButton:
    """Tests for _create_simple_icon_button method."""

    def test_create_simple_icon_button_object_name(self, mocked_insert_view):
        """Button has object name 'icon-button'."""
        callback = Mock()
        button = mocked_insert_view['insert_view']._create_simple_icon_button(
            callback,
            ":assets/icons/test.svg"
        )

        assert button.objectName() == "icon-button"

    def test_create_simple_icon_button_connects_on_click(self, mocked_insert_view, qtbot):
        """Clicking the button triggers the provided callback."""
        callback = Mock()
        button = mocked_insert_view['insert_view']._create_simple_icon_button(
            callback,
            ":assets/icons/test.svg"
        )
        qtbot.addWidget(button)

        button.click()

        callback.assert_called_once()


class TestActionMenu:
    """Tests for action menu functionality."""

    def test_menu_button_click_opens_action_menu(self, mocked_insert_view):
        """Clicking the menu button creates an ActionMenu and sets action_menu_open = True."""
        # Create a menu button
        menu_button = QPushButton()
        menu_button.setGeometry(0, 0, 40, 40)

        # Mock mapToGlobal to avoid Qt coordinate issues in tests
        menu_button.mapToGlobal = Mock(return_value=menu_button.rect().bottomLeft())

        mocked_insert_view['insert_view']._InsertView__action_menu_open_close(menu_button)

        assert mocked_insert_view['insert_view'].action_menu_open is True
        mocked_insert_view['action_menu_class'].assert_called_once_with(mocked_insert_view['insert_view'])
        assert mocked_insert_view['insert_view'].action_menu is not None

    def test_exit_action_resets_flag(self, mocked_insert_view, qtbot):
        """After aboutToHide fires, action_menu_open eventually resets to False."""
        menu_button = QPushButton()
        menu_button.setGeometry(0, 0, 40, 40)
        menu_button.mapToGlobal = Mock(return_value=menu_button.rect().bottomLeft())

        # Open the menu
        mocked_insert_view['insert_view']._InsertView__action_menu_open_close(menu_button)
        assert mocked_insert_view['insert_view'].action_menu_open is True

        # Simulate the menu being hidden (triggers __on_menu_hide via aboutToHide signal)
        mocked_insert_view['insert_view']._InsertView__on_menu_hide()

        # Flag should now be reset
        assert mocked_insert_view['insert_view'].action_menu_open is False