from typing import Callable

from PySide6.QtWidgets import QDialog


class DialogBase(QDialog):
    size: tuple[int, int]
    title: str
    on_close: Callable[[], None]

    def __init__(self, on_close: Callable[[], None], title: str, size: tuple[int, int] = (100, 100)):
        super().__init__()
        self.on_close = on_close
        self.size = size
        self.title = title

        self.setWindowTitle(f"{self.title}")
        self.setFixedSize(size[0], size[1])
        self.exec()
