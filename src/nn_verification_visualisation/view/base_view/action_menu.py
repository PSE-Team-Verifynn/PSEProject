from __future__ import annotations
import sys
from typing import List

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QWidget, QPushButton, QMenu, QGraphicsDropShadowEffect, QApplication, QMainWindow

from nn_verification_visualisation.view.dialogs.info_popup import InfoPopup
from nn_verification_visualisation.view.dialogs.info_type import InfoType

from typing import TYPE_CHECKING

from nn_verification_visualisation.view.dialogs.settings_dialog import SettingsDialog
from nn_verification_visualisation.model.data.storage import Storage

if TYPE_CHECKING:
    from nn_verification_visualisation.view.base_view.insert_view import InsertView


class ActionMenu(QWidget):

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

    def __exit_action(self):
        """
        Closes the main window of the application
        """
        app = QApplication.instance()
        main_window = None

        # Find the MainWindow instance
        for widget in app.topLevelWidgets():
            if isinstance(widget, QMainWindow):
                main_window = widget
                break

        if main_window:
            Storage.save_to_disk()
            main_window.close()
