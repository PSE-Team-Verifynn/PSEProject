from __future__ import annotations
import sys
from typing import List

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QWidget, QPushButton, QMenu, QGraphicsDropShadowEffect

from nn_verification_visualisation.view.base_view.action_menu_item import ActionMenuItem
from nn_verification_visualisation.view.dialogs.info_popup import InfoPopup
from nn_verification_visualisation.view.dialogs.info_type import InfoType

from typing import TYPE_CHECKING

from nn_verification_visualisation.view.dialogs.settings_dialog import SettingsDialog

if TYPE_CHECKING:
    from nn_verification_visualisation.view.base_view.insert_view import InsertView


class ActionMenu(QWidget):
    items: List[ActionMenuItem]

    def __init__(self, parent: InsertView):
        super().__init__()
        self.menu = QMenu()
        self.parent = parent
        self.menu.setWindowFlags(
            Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint | Qt.WindowType.NoDropShadowWindowHint
        )
        self.menu.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        settings_action = self.menu.addAction("Settings")
        settings_action.triggered.connect(self.__settings_action)

        exit_action = self.menu.addAction("Exit")
        exit_action.triggered.connect(self.__exit_action)

        shadow = QGraphicsDropShadowEffect(self.menu)
        shadow.setBlurRadius(10)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(0, 0, 0, 60))

        self.menu.setGraphicsEffect(shadow)

    def __settings_action(self):
        settings_dialog = SettingsDialog(self.parent.close_dialog)
        self.parent.open_dialog(settings_dialog)
        print("settings")

    def __exit_action(self):

        cancel_button = QPushButton("Cancel")
        cancel_button.setObjectName("light-button")
        confirm_button = QPushButton("Continue")
        confirm_button.setObjectName("error-button")
        confirm_button.clicked.connect(lambda: sys.exit())

        buttons = [cancel_button, confirm_button]
        text = "Do you really want to exit the program?"
        dialog = InfoPopup(self.parent.close_dialog, format(text), InfoType.WARNING, buttons)

        self.parent.open_dialog(dialog)