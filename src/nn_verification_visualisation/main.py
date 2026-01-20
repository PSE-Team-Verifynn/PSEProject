# This is a sample Python script.
import sys

from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from view.base_view.main_window import MainWindow

def main():
    app = QApplication(sys.argv)

    app.setStyle("Fusion")
    app.setPalette(app.style().standardPalette())

    window = MainWindow()

    window.show()

    sys.exit(app.exec())

if __name__ == '__main__':
    main()