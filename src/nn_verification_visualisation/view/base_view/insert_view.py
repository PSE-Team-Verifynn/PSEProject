from typing import List

from PySide6.QtWidgets import QWidget, QVBoxLayout, QStackedLayout

from view.base_view.action_menu import ActionMenu
from view.base_view.tabs import Tabs
from view.dialogs.dialog_base import DialogBase

class InsertView(QWidget):
    tabs: Tabs
    action_menu: ActionMenu
    page_layout: QVBoxLayout

    __dialog_stack: List[DialogBase]

    def __init__(self, tabs_closable: bool):
        super().__init__()

        self.tabs = Tabs(tabs_closable)

        self.container_layout = QStackedLayout()

        self.page_layout = QVBoxLayout()
        self.page_layout.addWidget(self.tabs)
        self.page_layout.setContentsMargins(0, 0, 0, 0)

        self.container = QWidget()
        self.container.setLayout(self.page_layout)

        self.container.setGeometry(self.rect())

        self.setLayout(self.container_layout)
        self.container_layout.addWidget(self.container)

        self.__dialog_stack = []

    def open_dialog(self, dialog: DialogBase):
        self.__dialog_stack.append(dialog)

        dialog.setParent(self)
        dialog.show()
        dialog.setGeometry(self.rect())

    def close_dialog(self) -> bool:
        if len(self.__dialog_stack) <= 0:
            return False

        self.__dialog_stack.pop().setParent(None)
        return True

    def resizeEvent(self, event):
        super().resizeEvent(event)
        for dialog in self.__dialog_stack:
            dialog.setGeometry(self.rect())
