from typing import List, Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton

from nn_verification_visualisation.view.dialogs.dialog_base import DialogBase
from nn_verification_visualisation.view.dialogs.info_type import InfoType

class InfoPopup(DialogBase):
    text: str
    info_type: InfoType
    buttons: List[QPushButton]

    styles = {
        InfoType.INFORMATION: "header-info",
        InfoType.CONFIRMATION: "header-success",
        InfoType.WARNING: "header-warning",
        InfoType.ERROR: "header-error"
    }

    titles = {
        InfoType.INFORMATION: "Information",
        InfoType.CONFIRMATION: "Success",
        InfoType.WARNING: "Warning!",
        InfoType.ERROR: "Error!"
    }

    def __init__(self, on_close: Callable[[], None], text: str, info_type: InfoType, buttons: List[QPushButton]=None):
        if buttons is None:
            buttons = []
        self.info_type = info_type
        self.buttons = buttons
        self.text = text
        super().__init__(on_close, InfoPopup.titles[info_type], (500, 150))

        self.header.setObjectName(InfoPopup.styles[info_type])


    def get_content(self) -> QWidget:
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container.setLayout(container_layout)

        label = QLabel(self.text)
        label.setWordWrap(True)
        label.setObjectName("popup-content")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        container_layout.addWidget(label)

        if len(self.buttons) > 0:
            button_bar = QHBoxLayout()
            for button in self.buttons:
                button.clicked.connect(self.on_close)
                button_bar.addWidget(button)
            container_layout.addLayout(button_bar)

        return container