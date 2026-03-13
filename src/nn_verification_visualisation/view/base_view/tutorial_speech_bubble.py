from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout


class TutorialSpeechBubble(QWidget):
    __margin: int = 15
    __spacing: int = 15
    def __init__(self, heading: str, text: str):
        super().__init__()

        layout = QVBoxLayout(self)

        container = QWidget()
        container.setObjectName("card")

        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(self.__margin,self.__margin,self.__margin,self.__margin)
        container_layout.setSpacing(self.__spacing)

        title = QLabel(heading)
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        label = QLabel(text)

        container_layout.addWidget(title)
        container_layout.addWidget(label)

        hbox = QHBoxLayout()
        hbox.addStretch()
        hbox.addWidget(container)
        hbox.addStretch()

        layout.addStretch()
        layout.addLayout(hbox)
        layout.addStretch()

