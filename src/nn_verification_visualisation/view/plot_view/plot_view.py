from PySide6.QtWidgets import QVBoxLayout, QLabel

from controller.input_manager.plot_view_controller import PlotViewController
from view.base_view.insert_view import InsertView
from view.dialogs.fullscreen_plot_dialog import FullscreenPlotDialog


class PlotView(InsertView):
    controller: PlotViewController

    def __init__(self):
        super().__init__(True)
        self.controller = PlotViewController(self)

        layout = QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(QLabel("Plot"))
        # dialog = FullscreenPlotDialog(None)