from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton
from PySide6.QtGui import QIcon

from nn_verification_visualisation.controller.input_manager.plot_view_controller import PlotViewController
from nn_verification_visualisation.view.base_view.insert_view import InsertView
from nn_verification_visualisation.view.plot_view.plot_page import PlotPage

class PlotView(InsertView):
    controller: PlotViewController

    def __init__(self, change_view: Callable[[], None], parent=None):
        super().__init__(parent)
        self.controller = PlotViewController(self)

        self.tabs.add_tab(PlotPage(self.controller))

        add_button = self._create_simple_icon_button(self.controller.open_plot_generation_dialog, ":assets/icons/add_icon.svg")

        view_toggle_button = QPushButton()
        view_toggle_button.clicked.connect(change_view)
        view_toggle_button.setObjectName("switch-button")
        view_toggle_button.setIcon(QIcon(":assets/icons/plot/switch.svg"))

        self.set_bar_corner_widgets([add_button, view_toggle_button],Qt.Corner.TopRightCorner, width=110)
