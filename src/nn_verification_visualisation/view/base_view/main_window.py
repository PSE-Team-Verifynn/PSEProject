from time import sleep

from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel, QStyleFactory
from PySide6.QtGui import QPalette, QColor
from PySide6.QtCore import Qt

from nn_verification_visualisation.view.base_view.base_view import BaseView
from nn_verification_visualisation.view.base_view.color_manager import ColorManager


class MainWindow(QMainWindow):
    base_view: BaseView
    __WINDOW_TITLE: str = "PSE Neuron App"

    def __init__(self, color_manager: ColorManager):
        super().__init__()

        self.setWindowTitle(self.__WINDOW_TITLE)
        self.resize(800, 600)

        # Central widget (QMainWindow *requires* one)
        base_view = BaseView(color_manager, self)
        self.setCentralWidget(base_view)
