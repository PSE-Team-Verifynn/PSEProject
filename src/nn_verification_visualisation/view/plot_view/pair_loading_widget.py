from typing import Callable

from PySide6.QtCore import QThread
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtWidgets import QPushButton, QLabel, QHBoxLayout, QFrame
from nn_verification_visualisation.view.plot_view.status import Status

class PairLoadingWidget(QFrame):
    '''
    List item that displays the status of a single running algorithm.
    Shows the name of the neuron pair, a status icon and a button.
    '''

    status: Status
    error: BaseException

    __on_click: Callable[[], None]
    __name: str

    __button: QPushButton
    __title: QLabel
    __icon: QSvgWidget

    def __init__(self, name: str, on_click: Callable[[], None] = None):
        '''
        :param name: name of the neuron pair, gets displayed
        :param on_click: function to call when the button gets pressed
        '''
        self.__on_click = on_click
        self.__name = name
        super().__init__()

        self.__button = QPushButton()
        self.__button.setVisible(False)
        self.__button.clicked.connect(self.__on_click)

        self.__title = QLabel(name)

        self.__icon = QSvgWidget()
        self.__icon.setFixedSize(20, 20)

        container_layout = QHBoxLayout()
        container_layout.addWidget(self.__title)
        container_layout.addWidget(self.__icon)
        container_layout.addStretch()
        container_layout.addWidget(self.__button)

        self.setObjectName("pair-loading-container")
        self.setFixedWidth(500)
        self.setFixedHeight(50)
        self.setLayout(container_layout)

    def set_status(self, status: Status):
        '''
        Function to internally and externally update the status of the widget.
        Gets called when the algorithm status updates.
        :param status: new status
        '''
        self.__button.setObjectName("")
        self.status = status

        match status:
            case Status.Ongoing:
                self.__button.setVisible(True)
                self.__button.setText("Cancel Execution")
                self.__icon.load(":assets/icons/hourglass.svg")
                status = "Loading"
            case Status.Done:
                self.__button.setVisible(False)
                status = "Completed"
                self.__icon.load(":assets/icons/check.svg")
            case Status.Failed:
                self.__button.setVisible(True)
                self.__button.setText("Show Error")
                self.__button.setObjectName("error-button")
                self.__icon.load(":assets/icons/error.svg")

                status = "Error"
        self.__title.setText("{} - {}".format(self.__name, status))

        self.__button.style().unpolish(self.__button)
        self.__button.style().polish(self.__button)
        self.__button.update()

