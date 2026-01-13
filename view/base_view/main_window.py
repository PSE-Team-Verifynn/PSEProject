from PySide6.QtWidgets import QMainWindow

from utils.singleton import SingletonMeta
from view.base_view.base_view import BaseView


class MainWindow(QMainWindow, metaclass=SingletonMeta):
    base_view: BaseView