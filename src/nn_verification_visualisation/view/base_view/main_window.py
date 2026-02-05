from time import sleep

from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel, QStyleFactory, QPushButton
from PySide6.QtGui import QPalette, QColor
from PySide6.QtCore import Qt

from nn_verification_visualisation.view.base_view.base_view import BaseView
from nn_verification_visualisation.view.base_view.color_manager import ColorManager
from nn_verification_visualisation.view.dialogs.info_popup import InfoPopup
from nn_verification_visualisation.view.dialogs.info_type import InfoType


class MainWindow(QMainWindow):
    base_view: BaseView
    __WINDOW_TITLE: str = "PSE Neuron App"

    def __init__(self, color_manager: ColorManager):
        super().__init__()

        self.exit_confirmed = False

        self.setWindowTitle(self.__WINDOW_TITLE)
        self.resize(800, 600)

        # Central widget (QMainWindow *requires* one)
        self.base_view = BaseView(color_manager, self)
        self.setCentralWidget(self.base_view)

    def closeEvent(self, event):
        if self.exit_confirmed:
            event.accept()
            return

        event.ignore()

        cancel_button = QPushButton("Cancel")
        cancel_button.setObjectName("light-button")
        cancel_button.clicked.connect(lambda: self.base_view.active_view.close_dialog())

        confirm_button = QPushButton("Continue")
        confirm_button.setObjectName("error-button")
        confirm_button.clicked.connect(lambda: self.__confirmed_exit())

        buttons = [cancel_button, confirm_button]
        text = "Do you really want to exit the program?"
        dialog = InfoPopup(self.base_view.active_view.close_dialog, format(text), InfoType.WARNING, buttons)

        self.base_view.active_view.open_dialog(dialog)

    def __confirmed_exit(self):
        self.exit_confirmed = True
        self.base_view.active_view.close_dialog()
        self.close()