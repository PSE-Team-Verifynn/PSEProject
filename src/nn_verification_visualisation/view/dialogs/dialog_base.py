from typing import Callable

from PySide6.QtWidgets import QWidget, QPushButton, QVBoxLayout, QLabel, QHBoxLayout
from PySide6.QtCore import Qt


class DialogBase(QWidget):
    size: tuple[int, int]
    title: str
    on_close: Callable[[], None]

    # abstract
    def get_content(self) -> QWidget:
        pass

    def __init__(self, on_close: Callable[[], None], title: str, size: tuple[int, int] | None = None):
        super().__init__()
        self.on_close = on_close
        self.size = size
        self.title = title

        self.setStyleSheet("background-color: rgba(0, 0, 0, 150);")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)  # important!

        self.dialog = QWidget()
        self.dialog.setStyleSheet("background-color: #fff; border-radius: 10px;")

        outer_layout = QVBoxLayout()
        outer_layout.addWidget(self.dialog, alignment=Qt.AlignmentFlag.AlignCenter)
        self.setLayout(outer_layout)

        layout = QVBoxLayout(self.dialog)
        layout.addWidget(self.get_title_bar())

        layout.addWidget(self.get_content())

        if size is not None:
            self.dialog.setFixedWidth(size[0])
            self.dialog.setFixedHeight(size[1])

    def get_title_bar(self) -> QWidget:
        bar = QWidget()

        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel(self.title)

        close_btn = QPushButton("X") # tmp
        close_btn.clicked.connect(self.on_close)

        bar_layout.addWidget(title, alignment=Qt.AlignmentFlag.AlignCenter)
        bar_layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignRight)

        return bar
