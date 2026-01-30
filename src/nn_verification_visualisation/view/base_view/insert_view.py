from typing import List, Callable

from PySide6.QtWidgets import QWidget, QVBoxLayout, QStackedLayout, QPushButton, QHBoxLayout, QMenu, \
    QGraphicsDropShadowEffect
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QIcon, QColor

from nn_verification_visualisation.view.base_view.action_menu import ActionMenu
from nn_verification_visualisation.view.base_view.tabs import Tabs
from nn_verification_visualisation.view.dialogs.dialog_base import DialogBase
from PySide6.QtCore import QSize


class InsertView(QWidget):
    tabs: Tabs
    action_menu: ActionMenu | None
    action_menu_open: bool = False
    page_layout: QVBoxLayout

    __dialog_stack: List[DialogBase]

    def __init__(self):
        super().__init__()

        self.tabs = Tabs(self.close_tab)

        self.container_layout = QStackedLayout()
        self.container_layout.setContentsMargins(0, 0, 0, 0)

        self.page_layout = QVBoxLayout()
        self.page_layout.addWidget(self.tabs)
        self.page_layout.setContentsMargins(0, 0, 0, 0)
        self.page_layout.setSpacing(0)

        self.container = QWidget()
        self.container.setLayout(self.page_layout)

        self.container.setGeometry(self.rect())

        self.setLayout(self.container_layout)
        self.container_layout.addWidget(self.container)

        self.__dialog_stack = []

        self.action_menu = None

        menu_button = self.set_bar_icon_button(lambda: self.__action_menu_open_close(menu_button), ":assets/icons/menu_icon.svg",
                                               Qt.Corner.TopLeftCorner)

    def set_bar_icon_button(self, on_click: Callable[[], None], icon: str, corner: Qt.Corner) -> QPushButton:
        button = QPushButton()
        button.setObjectName("icon-button")
        button.clicked.connect(on_click)

        button.setIcon(QIcon(icon))
        button.setFixedWidth(40)
        button.setFixedHeight(40)

        container = QWidget()
        container.sizeHint = lambda: QSize(button.width(), self.tabs.tabBar().height())
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch()
        layout.addWidget(button)
        layout.addStretch()

        self.tabs.setCornerWidget(container, corner)
        return button

    def close_tab(self, index: int):
        self.tabs.close_tab(index)

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

    def __action_menu_open_close(self, menu_button: QPushButton):
        if not self.action_menu_open:
            self.action_menu = ActionMenu(self)
            self.action_menu_open = True
            self.action_menu.menu.aboutToHide.connect(lambda: self.__exit_action())
            self.action_menu.menu.popup(menu_button.mapToGlobal(menu_button.rect().bottomLeft()))

    def __exit_action(self):
        self.action_menu.hide()
        QTimer.singleShot(200, lambda: setattr(self, "action_menu_open", False))