from PySide6.QtWidgets import QWidget

from view.base_view.insert_view import InsertView
from view.network_view.network_view import NetworkView
from view.plot_view.plot_view import PlotView


class BaseView(QWidget):
    active_view: InsertView
    plot_view: PlotView
    network_view: NetworkView