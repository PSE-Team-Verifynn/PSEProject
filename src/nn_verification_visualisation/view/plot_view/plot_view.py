from PySide6.QtWidgets import QVBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Qt

from nn_verification_visualisation.controller.input_manager.plot_view_controller import PlotViewController
from nn_verification_visualisation.view.base_view.insert_view import InsertView
from nn_verification_visualisation.view.dialogs.fullscreen_plot_dialog import FullscreenPlotDialog
from nn_verification_visualisation.view.plot_view.plot_page import PlotPage


class PlotView(InsertView):
    controller: PlotViewController

    def __init__(self):
        super().__init__(True)
        self.controller = PlotViewController(self)

        self.tabs.add_tab(PlotPage(None))

        self.set_bar_icon_button(lambda: None, ":assets/icons/menu_icon.svg", Qt.Corner.TopLeftCorner)
        self.set_bar_icon_button(self.controller.open_plot_generation_dialog, ":assets/icons/add_icon.svg", Qt.Corner.TopRightCorner)
