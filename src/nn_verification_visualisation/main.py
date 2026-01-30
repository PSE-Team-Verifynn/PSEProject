import sys
from pathlib import Path

from nn_verification_visualisation import resources_rc

from PySide6.QtWidgets import QApplication

from nn_verification_visualisation.view.base_view.color_manager import ColorManager
from nn_verification_visualisation.view.base_view.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    color_manager = ColorManager(app)
    style_path = Path(__file__).resolve().parent / "style.qss"
    color_manager.load_raw(str(style_path))

    window = MainWindow(color_manager)

    window.showMaximized()

    sys.exit(app.exec())

if __name__ == '__main__':
    main()
