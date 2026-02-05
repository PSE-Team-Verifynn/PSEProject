from typing import List, Callable

from PySide6.QtWidgets import QWidget, QVBoxLayout, QStackedLayout, QPushButton, QHBoxLayout, QMenu, \
    QGraphicsDropShadowEffect, QSizePolicy
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

    # Stack data structure that stores the current open dialogs (highest item is in front)
    __dialog_stack: List[DialogBase]

    def __init__(self, parent=None):
        super().__init__(parent)

        self.tabs = Tabs(self.close_tab, empty_page=self.get_default_tab())

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

        menu_button = self._create_simple_icon_button(lambda: self.__action_menu_open_close(menu_button),
                                               ":assets/icons/menu_icon.svg",)

        self.set_bar_corner_widgets([menu_button], Qt.Corner.TopLeftCorner)

    # override
    def get_default_tab(self) -> QWidget | None:
        return None

    def _create_simple_icon_button(self, on_click: Callable[[], None], icon: str) -> QPushButton:
        '''
        Creates a simple icon button with default parameters.
        :param on_click: function to be called on click
        :param icon: the asset path of the icon
        :return: the newly created button
        '''
        button = QPushButton()
        button.setObjectName("icon-button")
        button.clicked.connect(on_click)

        button.setIcon(QIcon(icon))
        return button

    def set_bar_corner_widgets(self, widgets: List[QWidget], corner: Qt.Corner, width: int = 40):
        '''
        Adds a list of QWidgets to a corner of the TabBar.
        :param width: width of the widget list
        :param corner: position of the button
        :return: the new button
        '''

        container = QWidget()
        container.sizeHint = lambda: QSize(width, self.tabs.tabBar().height())
        container.setFixedHeight(self.tabs.tabBar().sizeHint().height())
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch()
        [layout.addWidget(w, alignment=Qt.AlignmentFlag.AlignVCenter) for w in widgets]
        layout.addStretch()

        self.tabs.setCornerWidget(container, corner)

    def close_tab(self, index: int):
        self.tabs.close_tab(index)

    def open_dialog(self, dialog: DialogBase):
        '''
        Opens a new dialog
        :param dialog: dialog to be opened
        '''
        self.__dialog_stack.append(dialog)

        dialog.setParent(self)
        dialog.setVisible(True)
        dialog.setGeometry(self.rect())

    def close_dialog(self) -> bool:
        '''
        Closes the current dialog by removing it from the stack
        :return: if the removal was successful
        '''
        if len(self.__dialog_stack) <= 0:
            return False

        self.__dialog_stack.pop().setParent(None)
        return True

    # internal Qt-function to get notified on resize
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
