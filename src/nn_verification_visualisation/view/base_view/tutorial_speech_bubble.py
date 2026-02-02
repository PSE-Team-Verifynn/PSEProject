from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout


class TutorialSpeechBubble(QWidget):
    def __init__(self, heading: str, text: str):
        super().__init__()

        layout = QVBoxLayout(self)

        container = QWidget()
        container.setObjectName("card")

        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(15,15,15,15)
        container_layout.setSpacing(15)

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

