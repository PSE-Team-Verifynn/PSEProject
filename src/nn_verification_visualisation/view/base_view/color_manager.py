from PySide6.QtWidgets import QApplication

from PySide6.QtGui import QColor, QPalette
from PySide6.QtCore import Qt, QFile, QIODevice


class ColorManager:
    '''
    Manages the theme switch between the network view and the plot view. Stores the color and font data for both themes.
    Works by reading the stylesheet from disk and replacing its colors with the colors of the active theme.
    Sets a new stylesheet at every theme change.
    '''
    app: QApplication
    raw_stylesheet: str

    NETWORK_COLORS = {
        "bgt": "#000000",
        "ht": "#8f8f8f",
        "fgt": "#ffffff",
        "bg0": "#ffffff",
        "bg1": "#E5F0D2",
        "hbg1": "#f0f7e4",
        "bg2": "#C3D6A1",
        "hbg2": "#d1e0b8",
        "fg0": "#5E7A2D",
        "fg1": "#7C964D",
        "hfg1": "#87a353",
        "c0": "#CFCFD1",
        "success": "#89C4A3",
        "warning": "#E5BE6B",
        "error": "#E56B6F",
        "herror": "#f57d81",
        "component": "#CFCFD1"
    }

    DIAGRAM_COLORS = {
        "bgt": "#000000",
        "ht": "#8f8f8f",
        "fgt": "#ffffff",
        "bg0": "#ffffff",
        "bg1": "#E8EAED",
        "hbg1": "#f2f4f7",
        "bg2": "#CDD6DD",
        "hbg2": "#dde6ed",
        "fg0": "#436680",
        "fg1": "#6C879B",
        "hfg1": "#7895ab",
        "c0": "#CFCFD1",
        "success": "#89C4A3",
        "warning": "#E5BE6B",
        "error": "#E56B6F",
        "herror": "#f57d81",
        "component": "#CFCFD1"
    }

    def load_raw(self, path_str: str):
        file = QFile(path_str)
        file.open(QIODevice.ReadOnly | QIODevice.Text)
        self.raw_stylesheet = file.readAll().data().decode("utf-8")
        file.close()

    def set_colors(self, colors: dict[str, str]):
        '''
        Changes the color theme of the application. A new stylesheet gets created from self.raw_stylesheet by replacing its colors.
        The QPalette of the app is changed to update the colors of all default QWidgets.
        :param colors:
        :return:
        '''

        # Replace colors of style sheet
        stylesheet = self.raw_stylesheet
        for (key, val) in colors.items():
            stylesheet = stylesheet.replace("@" + key, val)

        # Create new QPalette
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

        # Update app
        self.app.setPalette(palette)
        self.app.setStyleSheet(stylesheet)

    def __init__(self, app: QApplication):
        self.app = app
