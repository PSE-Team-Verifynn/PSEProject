import sys

import resources_rc

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from nn_verification_visualisation.view.base_view.main_window import MainWindow

colors = {
    "bgt": "#000000",
    "ht" : "#8f8f8f",
    "fgt": "#ffffff",
    "bg0": "#ffffff",
    "bg1": "#E8EAED",
    "bg2": "#CDD6DD",
    "fg0": "#436680",
    "fg1": "#6C879B",
    "c0" : "#CFCFD1",
    "success" : "#89C4A3",
    "warning" : "#E58E6B",
    "error" : "#E56B6F",
}

def main():
    app = QApplication(sys.argv)

    with open("style.qss", "r") as f:
        stylesheet = f.read()

        for (key, val) in colors.items():
            stylesheet = stylesheet.replace("@" + key, val)

        app.setStyleSheet(stylesheet)

    # app.setStyle("Fusion")

    palette = QPalette()

    palette.setColor(QPalette.Window, QColor(colors["bg0"]))
    palette.setColor(QPalette.WindowText, Qt.GlobalColor.black)
    palette.setColor(QPalette.Base, QColor(colors["bg0"]))
    palette.setColor(QPalette.AlternateBase, colors["bg1"])
    palette.setColor(QPalette.ToolTipBase, Qt.GlobalColor.white)
    palette.setColor(QPalette.ToolTipText, Qt.GlobalColor.black)
    palette.setColor(QPalette.Text, colors["bgt"])
    palette.setColor(QPalette.Button, colors["fg0"])
    palette.setColor(QPalette.ButtonText,  colors["bgt"])
    palette.setColor(QPalette.BrightText, Qt.GlobalColor.red)
    palette.setColor(QPalette.Highlight, QColor("#0078d7"))
    palette.setColor(QPalette.HighlightedText, Qt.GlobalColor.white)

    app.setPalette(palette)

    window = MainWindow()

    window.showMaximized()

    sys.exit(app.exec())

if __name__ == '__main__':
    main()