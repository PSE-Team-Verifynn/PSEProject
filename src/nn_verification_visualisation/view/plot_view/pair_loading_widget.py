from typing import Callable, Any

from PySide6.QtWidgets import QWidget
from nn_verification_visualisation.view.plot_view.status import Status

class PairLoadingWidget(QWidget):
    name: str
    status: Status
    index: int
    on_click: Callable[[], Any]