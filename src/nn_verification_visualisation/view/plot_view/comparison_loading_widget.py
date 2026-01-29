from typing import List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea

from nn_verification_visualisation.view.plot_view.pair_loading_widget import PairLoadingWidget


class ComparisonLoadingWidget(QWidget):
    loaders: List[PairLoadingWidget]
    title: QWidget

    def __init__(self):
        super().__init__()

        self.title = QLabel("Loading...")
        self.title.setObjectName("title")

        self.loaders = self.__getPairList()

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        pair_container = QWidget()

        pair_layout = QVBoxLayout(pair_container)

        pair_layout.addStretch()

        for loader in self.loaders:
            pair_layout.addWidget(loader, alignment=Qt.AlignmentFlag.AlignHCenter)

        pair_layout.addStretch()
        scroll_area.setWidget(pair_container)

        layout = QVBoxLayout()
        layout.addWidget(self.title)
        layout.addWidget(scroll_area)

        self.setLayout(layout)

    def __getPairList(self) -> List[PairLoadingWidget]:
        loaders: List[PairLoadingWidget] = []
        for i in range(4):
            loaders.append(PairLoadingWidget("Neuron Pair {}".format(i + 1)))
            pass
        return loaders

    def add_plot(self):
        pass
