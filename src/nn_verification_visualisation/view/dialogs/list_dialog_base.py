from typing import Generic, TypeVar, List, Callable

from PySide6.QtWidgets import QWidget, QListWidget, QVBoxLayout, QHBoxLayout, QPushButton
from view.dialogs.dialog_base import DialogBase

T = TypeVar('T')

class ListDialogBase(Generic[T], DialogBase):
    data: List[T]
    list_widget: QListWidget
    has_edit: bool

    def __init__(self, on_close: Callable[[], None], title: str, data: List[T], has_edit: bool = False):
        self.data = data
        self.has_edit = has_edit
        super().__init__(on_close, title)

    # abstract
    def get_title(self, item: T) -> str:
        pass

    # abstract
    def on_add_clicked(self) -> None:
        pass

    # abstract
    def on_remove_clicked(self, item: T, index: int) -> bool:
        pass

    # abstract
    def on_edit_clicked(self, item: T) -> T | None:
        pass

    # abstract
    def on_confirm_clicked(self):
        pass

    def get_content(self) -> QWidget:

        content = QWidget()
        content_layout = QVBoxLayout(content)

        self.list_widget = QListWidget(self)

        for item in self.data:
            self.list_widget.addItem(self.get_title(item))

        self.list_widget.show()

        content_layout.addWidget(self.list_widget)

        button_bar_layout = QHBoxLayout()

        add_button = QPushButton("Add")
        add_button.clicked.connect(self.on_add_clicked)
        button_bar_layout.addWidget(add_button)

        if self.has_edit:
            edit_button = QPushButton("Edit")
            edit_button.clicked.connect(self.on_edit_clicked)
            button_bar_layout.addWidget(edit_button)

        remove_button = QPushButton("Remove")
        remove_button.clicked.connect(self.__internal_on_remove_clicked)
        button_bar_layout.addWidget(remove_button)

        confirm_button = QPushButton("Confirm")
        confirm_button.clicked.connect(self.on_confirm_clicked)
        button_bar_layout.addWidget(confirm_button)

        content_layout.addLayout(button_bar_layout)
        return content

    def add_item(self, item: T):
        self.data.append(item)
        self.list_widget.addItem(self.get_title(item))
        self.list_widget.show()

    def __internal_on_remove_clicked(self) -> None:
        list_items = self.list_widget.selectedItems()
        list_indices = self.list_widget.selectedIndexes()
        if not list_items or len(list_items) != 1:
            return
        item = list_items[0]
        index = list_indices[0].row()

        success = self.on_remove_clicked(self.data[index], index)
        if success:
            self.list_widget.takeItem(self.list_widget.row(item))
            self.data.pop(index)
            self.list_widget.show()

