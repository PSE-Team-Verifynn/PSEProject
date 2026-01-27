from PySide6.QtWidgets import QApplication
from pathlib import Path

from PySide6.QtGui import QColor, QPalette
from PySide6.QtCore import Qt


class ColorManager:
    app: QApplication
    raw_stylesheet: str

    NETWORK_COLORS = {
        "bgt": "#000000",
        "ht": "#8f8f8f",
        "fgt": "#ffffff",
        "bg0": "#ffffff",
        "bg1": "#E5F0D2",
        "bg2": "#C3D6A1",
        "fg0": "#5E7A2D",
        "fg1": "#7C964D",
        "hfg1": "#87a353",
        "c0": "#CFCFD1",
        "success": "#89C4A3",
        "warning": "#E58E6B",
        "error": "#E56B6F",
    }

    DIAGRAM_COLORS = {
        "bgt": "#000000",
        "ht": "#8f8f8f",
        "fgt": "#ffffff",
        "bg0": "#ffffff",
        "bg1": "#E8EAED",
        "bg2": "#CDD6DD",
        "fg0": "#436680",
        "fg1": "#6C879B",
        "hfg1": "#7895ab",
        "c0": "#CFCFD1",
        "success": "#89C4A3",
        "warning": "#E58E6B",
        "error": "#E56B6F",
    }

    def load_raw(self, path_str: str):
        self.raw_stylesheet = ((Path(__file__) / path_str).read_text())

    def set_colors(self, colors: dict[str, str]):
        stylesheet = self.raw_stylesheet
        for (key, val) in colors.items():
            stylesheet = stylesheet.replace("@" + key, val)

        palette = QPalette()

        palette.setColor(QPalette.Window, QColor(colors["bg0"]))
        palette.setColor(QPalette.WindowText, Qt.GlobalColor.black)
        palette.setColor(QPalette.Base, QColor(colors["bg0"]))
        palette.setColor(QPalette.AlternateBase, colors["bg1"])
        palette.setColor(QPalette.ToolTipBase, Qt.GlobalColor.white)
        palette.setColor(QPalette.ToolTipText, Qt.GlobalColor.black)
        palette.setColor(QPalette.Text, colors["bgt"])
        palette.setColor(QPalette.Button, colors["fg0"])
        palette.setColor(QPalette.ButtonText, colors["bgt"])
        palette.setColor(QPalette.BrightText, Qt.GlobalColor.red)
        palette.setColor(QPalette.Highlight, QColor("#0078d7"))
        palette.setColor(QPalette.HighlightedText, Qt.GlobalColor.white)

        self.app.setPalette(palette)
        self.app.setStyleSheet(stylesheet)

    def __init__(self, app: QApplication):
        self.app = app
