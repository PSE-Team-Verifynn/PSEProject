from typing import Generic, TypeVar, List, Callable

from PySide6.QtWidgets import QWidget, QListWidget, QVBoxLayout, QHBoxLayout, QPushButton
from view.dialogs.dialog_base import DialogBase

T = TypeVar('T')

class ListDialogBase(Generic[T], DialogBase):
    data: List[T]
    on_click: Callable[[T], None]
    on_change: Callable[[T], None]
    on_add: Callable[[T], None]
    on_remove: Callable[[T], None]

    def __init__(self, on_click: Callable[[T], None], title: str, data: List[T]):
        super().__init__(on_click, title)
        self.data = data

    # abstract
    def get_title(self, item: T) -> str:
        pass

    # abstract
    def on_add_clicked(self) -> T | None:
        pass

    # abstract
    def on_remove_clicked(self, item: T) -> None:
        pass

    # abstract
    def on_edit_clicked(self, item: T) -> T | None:
        pass

    def get_content(self) -> QWidget:

        content = QWidget()
        content_layout = QVBoxLayout(content)

        list_widget = QListWidget(self)

        items = ["Apple", "Banana", "Cherry"]

        for item in items:
            list_widget.addItem(str(item))
            list_widget.show()

        content_layout.addWidget(list_widget)

        button_bar_layout = QHBoxLayout()

        add_button = QPushButton("Add")
        add_button.clicked.connect(self.on_add_clicked)
        button_bar_layout.addWidget(add_button)

        edit_button = QPushButton("Edit")
        edit_button.clicked.connect(self.on_edit_clicked)
        button_bar_layout.addWidget(edit_button)

        remove_button = QPushButton("Remove")
        button_bar_layout.addWidget(remove_button)

        content_layout.addLayout(button_bar_layout)
        return content