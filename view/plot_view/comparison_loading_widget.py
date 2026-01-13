from typing import List

from PySide6.QtWidgets import QWidget

from model.data.plot import Plot
from view.plot_view.pair_loading_widget import PairLoadingWidget


class ComparisonLoadingWidget(QWidget):
    loaders: List[PairLoadingWidget]

    def add_plot(self, plot: Plot):
        pass