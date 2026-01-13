from typing import List

from PySide6.QtWidgets import QWidget

from view.plot_view.plot_widget import PlotWidget


class PlotPage(QWidget):
    plots: List[PlotWidget]
    locked: List[PlotWidget]

    def change_lock(self, widget:PlotWidget):
        pass

    def fullscreen(self, widget:PlotWidget):
        pass

    def export(self, widget:PlotWidget):
        pass

    def transform_changed(self, widget:PlotWidget, transform: tuple[float, float, float, float]):
        pass