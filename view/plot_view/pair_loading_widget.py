from typing import Callable

from PySide6.QtWidgets import QWidget
from view.plot_view.status import Status

class PairLoadingWidget(QWidget):
    name: str
    status: Status
    index: int
    on_click: Callable[[], None]