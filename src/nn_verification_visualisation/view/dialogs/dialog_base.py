from typing import Callable

from PySide6.QtGui import QColor, QIcon
from PySide6.QtWidgets import QWidget, QPushButton, QVBoxLayout, QLabel, QHBoxLayout, QSizePolicy
from PySide6.QtCore import Qt, QSize

import os

class DialogBase(QWidget):
    size: tuple[int, int]
    title: str
    on_close: Callable[[], None]

    # abstract
    def get_content(self) -> QWidget:
        pass

    def __init__(self, on_close: Callable[[], None], title: str, size: tuple[int, int] | None = None, has_title: bool = True):
        super().__init__()
        self.on_close = on_close
        self.size = size
        self.title = title

        self.setObjectName("faded-background")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)  # important!

        self.dialog = QWidget()
        self.dialog.setObjectName("dialog-pane")

        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(0,0,0,0)
        outer_layout.addWidget(self.dialog, alignment=Qt.AlignmentFlag.AlignCenter)
        self.setLayout(outer_layout)

        self.header = self.get_title_bar()
        self.header.setObjectName("header")

        layout = QVBoxLayout(self.dialog)
        layout.setContentsMargins(0,0,0,0)
        if has_title:
            layout.addWidget(self.header)
        layout.addStretch()

        layout.addWidget(self.get_content())

        if size is not None:
            self.dialog.setSizePolicy(
                QSizePolicy.Policy.Preferred,
                QSizePolicy.Policy.Preferred
            )
            self.dialog.sizeHint = lambda: QSize(size[0], size[1])

    def get_title_bar(self) -> QWidget:
        bar = QWidget()

        button_size = 35

        left_padding = QWidget()
        left_padding.setFixedWidth(button_size)

        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel(self.title)
        title.setObjectName("subtitle")
        title.setAlignment(Qt.AlignCenter)

        close_btn = QPushButton() # tmp
        close_btn.setFixedWidth(button_size)
        close_btn.setFixedHeight(button_size)
        close_btn.setObjectName("close-button")
        close_btn.setIcon(QIcon(":assets/icons/close_icon.svg"))
        close_btn.clicked.connect(self.on_close)

        bar_layout.addWidget(left_padding)
        bar_layout.addStretch()
        bar_layout.addWidget(title, alignment=Qt.AlignmentFlag.AlignCenter)
        bar_layout.addStretch()
        bar_layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignRight)

        return bar
