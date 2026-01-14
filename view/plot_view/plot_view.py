from PySide6.QtWidgets import QVBoxLayout, QLabel

from view.base_view.insert_view import InsertView
from view.dialogs.fullscreen_plot_dialog import FullscreenPlotDialog


class PlotView(InsertView):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(QLabel("Plot"))
        dialog = FullscreenPlotDialog(None)