from typing import Callable

from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QFrame, QScrollArea
from nn_verification_visualisation.view.plot_view.status import Status


class PairLoadingWidget(QFrame):
    __on_click: Callable[[], None]
    __name: str

    __button: QPushButton
    __title: QLabel

    def __init__(self, name: str, on_click: Callable[[], None] = None):
        self.__on_click = on_click
        self.__name = name
        super().__init__()

        self.__button = QPushButton()
        self.__button.setVisible(False)

        self.__title = QLabel(name)

        container_layout = QHBoxLayout()
        container_layout.addWidget(self.__title)
        container_layout.addStretch()
        container_layout.addWidget(self.__button)

        self.setObjectName("pair-loading-container")
        self.setFixedWidth(400)
        self.setLayout(container_layout)

        self.set_status(Status.Ongoing)

    def set_status(self, status: Status):
        match status:
            case Status.Ongoing:
                self.__button.setVisible(True)
                self.__button.setText("Cancel Execution")
                status = "Loading"
            case Status.Done:
                self.__button.setVisible(False)
                status = "Completed"
            case Status.Failed:
                self.__button.setVisible(True)
                self.__button.setText("Show Error")
                status = "Error"
        self.__title.setText("{} - {}".format(self.__name, status))
        pass
