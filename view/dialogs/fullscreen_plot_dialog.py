from PySide6.QtWidgets import QVBoxLayout

from view.dialogs.dialog_base import DialogBase
from model.data.plot import Plot

class FullscreenPlotDialog(DialogBase):
    plot: Plot

    def __init__(self, plot: Plot):
        self.plot = plot
        super().__init__(lambda: None, title="Fullscreen Plot")

    