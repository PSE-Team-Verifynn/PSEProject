from PySide6.QtCore import Qt
from sympy.codegen.ast import none

from nn_verification_visualisation.controller.input_manager.plot_view_controller import PlotViewController
from nn_verification_visualisation.view.base_view.insert_view import InsertView
from nn_verification_visualisation.view.plot_view.plot_page import PlotPage


class PlotView(InsertView):
    controller: PlotViewController

    def __init__(self, parent=None, polygons: list[list[tuple[float, float]]] = None):
        super().__init__(parent)
        self.controller = PlotViewController(self)

        self.tabs.add_tab(PlotPage(self.controller, polygons))

        self.set_bar_icon_button(self.controller.open_plot_generation_dialog, ":assets/icons/add_icon.svg", Qt.Corner.TopRightCorner)

    def add_plot_tab(self, polygons:list[list[tuple[float, float]]] ):
        '''
        Adds polygons tab to the QTabWidget. Only updates UI, not the backend.
        :param polygons: Data object of the new tab.
        '''
        self.tabs.add_tab(PlotPage(self.controller, polygons))