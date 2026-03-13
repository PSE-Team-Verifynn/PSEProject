from unittest.mock import Mock

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QWidget, QLabel, QListWidget, QPushButton

from nn_verification_visualisation.view.dialogs.dialog_base import DialogBase
from nn_verification_visualisation.view.dialogs.list_dialog_base import ListDialogBase
from nn_verification_visualisation.view.dialogs.sample_results_dialog import SampleResultsDialog


class _SimpleDialog(DialogBase):
    def get_content(self) -> QWidget:
        return QLabel("content")


class _SimpleListDialog(ListDialogBase[str]):
    def __init__(self, on_close, data, has_edit=False):
        self.add_clicked = Mock()
        self.remove_clicked = Mock(return_value=True)
        self.edit_clicked = Mock()
        self.confirm_clicked = Mock()
        super().__init__(on_close, "List Title", data, has_edit)

    def get_title(self, item: str) -> str:
        return item.upper()

    def on_add_clicked(self) -> None:
        self.add_clicked()

    def on_remove_clicked(self, item: str, index: int) -> bool:
        return self.remove_clicked(item, index)

    def on_edit_clicked(self, item: str) -> None:
        self.edit_clicked(item)

    def on_confirm_clicked(self):
        self.confirm_clicked()


def test_dialog_base_builds_header_and_content(qapp):
    on_close = Mock()

    dialog = _SimpleDialog(on_close, "Dialog Title", (320, 180))

    assert dialog.title == "Dialog Title"
    assert dialog.header is not None
    assert dialog.dialog is not None
    assert dialog.dialog.sizeHint().width() == 320
    assert dialog.dialog.sizeHint().height() == 180


def test_dialog_base_escape_key_triggers_close(qapp):
    on_close = Mock()
    dialog = _SimpleDialog(on_close, "Dialog Title")

    dialog.keyPressEvent(QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier))

    on_close.assert_called_once()


def test_list_dialog_base_populates_and_adds_items(qapp):
    dialog = _SimpleListDialog(Mock(), ["first", "second"])

    assert dialog.list_widget.count() == 2
    assert dialog.list_widget.item(0).text() == "FIRST"

    dialog.add_item("third")

    assert dialog.data == ["first", "second", "third"]
    assert dialog.list_widget.item(2).text() == "THIRD"


def test_list_dialog_base_remove_selected_item_updates_data_and_widget(qapp):
    dialog = _SimpleListDialog(Mock(), ["first", "second"])
    dialog.list_widget.setCurrentRow(1)

    dialog._ListDialogBase__internal_on_remove_clicked()

    dialog.remove_clicked.assert_called_once_with("second", 1)
    assert dialog.data == ["first"]
    assert dialog.list_widget.count() == 1


def test_list_dialog_base_edit_selected_item_delegates(qapp):
    dialog = _SimpleListDialog(Mock(), ["first", "second"], has_edit=True)
    dialog.list_widget.setCurrentRow(0)

    dialog._ListDialogBase__internal_on_edit_clicked()

    dialog.edit_clicked.assert_called_once_with("first")


def test_sample_results_dialog_builds_metrics_widget(qapp):
    result = {"num_samples": 1, "metrics": [], "outputs": []}

    dialog = SampleResultsDialog(Mock(), result)
    content = dialog.get_content()

    assert dialog.result is result
    assert content.layout() is not None
    assert content.layout().count() == 1
