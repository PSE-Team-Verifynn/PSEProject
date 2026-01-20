from PySide6.QtWidgets import QVBoxLayout, QLabel, QPushButton

from nn_verification_visualisation.controller.input_manager.plot_view_controller import PlotViewController
from nn_verification_visualisation.view.base_view.insert_view import InsertView
from nn_verification_visualisation.view.dialogs.fullscreen_plot_dialog import FullscreenPlotDialog
from nn_verification_visualisation.view.plot_view.plot_page import PlotPage


class PlotView(InsertView):
    controller: PlotViewController

    def __init__(self):
        super().__init__(True)
        self.controller = PlotViewController(self)

        layout = QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(QLabel("Plot"))

        self.button = QPushButton("Create New Diagram", self)
        self.button.move(100, 80)
        self.button.clicked.connect(self.controller.open_plot_generation_dialog)

        self.page_layout.addWidget(self.button)

        self.tabs.add_tab(PlotPage(None))
