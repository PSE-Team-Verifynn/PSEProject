from unittest.mock import MagicMock
from PySide6.QtWidgets import QPushButton, QLabel
from PySide6.QtCore import Qt

from nn_verification_visualisation.view.dialogs.info_popup import InfoPopup
from nn_verification_visualisation.view.dialogs.info_type import InfoType


class TestInfoPopupInit:
    def test_information_type_sets_correct_title(self, qtbot):
        popup = InfoPopup(MagicMock(), "Some info", InfoType.INFORMATION)
        qtbot.addWidget(popup)
        assert popup.title == "Information"

    def test_confirmation_type_sets_correct_title(self, qtbot):
        popup = InfoPopup(MagicMock(), "All good", InfoType.CONFIRMATION)
        qtbot.addWidget(popup)
        assert popup.title == "Success"

    def test_warning_type_sets_correct_title(self, qtbot):
        popup = InfoPopup(MagicMock(), "Be careful", InfoType.WARNING)
        qtbot.addWidget(popup)
        assert popup.title == "Warning!"

    def test_error_type_sets_correct_title(self, qtbot):
        popup = InfoPopup(MagicMock(), "Something broke", InfoType.ERROR)
        qtbot.addWidget(popup)
        assert popup.title == "Error!"

    def test_stores_text(self, qtbot):
        popup = InfoPopup(MagicMock(), "Hello world", InfoType.INFORMATION)
        qtbot.addWidget(popup)
        assert popup.text == "Hello world"

    def test_stores_info_type(self, qtbot):
        popup = InfoPopup(MagicMock(), "msg", InfoType.WARNING)
        qtbot.addWidget(popup)
        assert popup.info_type == InfoType.WARNING

    def test_buttons_defaults_to_empty_list(self, qtbot):
        popup = InfoPopup(MagicMock(), "msg", InfoType.INFORMATION)
        qtbot.addWidget(popup)
        assert popup.buttons == []

    def test_custom_buttons_stored(self, qtbot):
        btn = QPushButton("OK")
        popup = InfoPopup(MagicMock(), "msg", InfoType.INFORMATION, buttons=[btn])
        qtbot.addWidget(popup)
        assert popup.buttons == [btn]

    def test_header_style_information(self, qtbot):
        popup = InfoPopup(MagicMock(), "msg", InfoType.INFORMATION)
        qtbot.addWidget(popup)
        assert popup.header.objectName() == "header-info"

    def test_header_style_confirmation(self, qtbot):
        popup = InfoPopup(MagicMock(), "msg", InfoType.CONFIRMATION)
        qtbot.addWidget(popup)
        assert popup.header.objectName() == "header-success"

    def test_header_style_warning(self, qtbot):
        popup = InfoPopup(MagicMock(), "msg", InfoType.WARNING)
        qtbot.addWidget(popup)
        assert popup.header.objectName() == "header-warning"

    def test_header_style_error(self, qtbot):
        popup = InfoPopup(MagicMock(), "msg", InfoType.ERROR)
        qtbot.addWidget(popup)
        assert popup.header.objectName() == "header-error"


class TestInfoPopupGetContent:
    def _find_label(self, widget):
        """Recursively find a QLabel in a widget's children."""
        for child in widget.findChildren(QLabel):
            return child
        return None

    def test_content_label_shows_text(self, qtbot):
        popup = InfoPopup(MagicMock(), "Test message", InfoType.INFORMATION)
        qtbot.addWidget(popup)
        content = popup.get_content()
        label = self._find_label(content)
        assert label is not None
        assert label.text() == "Test message"

    def test_content_label_has_correct_object_name(self, qtbot):
        popup = InfoPopup(MagicMock(), "msg", InfoType.INFORMATION)
        qtbot.addWidget(popup)
        content = popup.get_content()
        label = self._find_label(content)
        assert label.objectName() == "popup-content"

    def test_content_label_is_word_wrapped(self, qtbot):
        popup = InfoPopup(MagicMock(), "msg", InfoType.INFORMATION)
        qtbot.addWidget(popup)
        content = popup.get_content()
        label = self._find_label(content)
        assert label.wordWrap() is True

    def test_content_label_is_centered(self, qtbot):
        popup = InfoPopup(MagicMock(), "msg", InfoType.INFORMATION)
        qtbot.addWidget(popup)
        content = popup.get_content()
        label = self._find_label(content)
        assert label.alignment() == Qt.AlignmentFlag.AlignCenter

    def test_no_buttons_means_no_button_widgets(self, qtbot):
        popup = InfoPopup(MagicMock(), "msg", InfoType.INFORMATION)
        qtbot.addWidget(popup)
        content = popup.get_content()
        buttons = content.findChildren(QPushButton)
        assert len(buttons) == 0

    def test_buttons_are_added_to_content(self, qtbot):
        btn1 = QPushButton("Yes")
        btn2 = QPushButton("No")
        popup = InfoPopup(MagicMock(), "msg", InfoType.CONFIRMATION, buttons=[btn1, btn2])
        qtbot.addWidget(popup)
        content = popup.get_content()
        buttons = content.findChildren(QPushButton)
        assert btn1 in buttons
        assert btn2 in buttons

    def test_button_click_calls_on_close(self, qtbot):
        on_close = MagicMock()
        btn = QPushButton("OK")
        popup = InfoPopup(on_close, "msg", InfoType.INFORMATION, buttons=[btn])
        qtbot.addWidget(popup)
        btn.click()
        on_close.assert_called()