from PySide6.QtWidgets import QVBoxLayout, QLabel

from controller.input_manager.plot_view_controller import PlotViewController
from view.base_view.insert_view import InsertView


class PlotView(InsertView):
    controller: PlotViewController

    def __init__(self):
        super().__init__()
        self.controller = PlotViewController(self)

        layout = QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(QLabel("Plot"))