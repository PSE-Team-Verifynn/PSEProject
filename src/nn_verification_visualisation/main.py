import sys
import resources_rc

from PySide6.QtWidgets import QApplication

from nn_verification_visualisation.view.base_view.color_manager import ColorManager
from nn_verification_visualisation.view.base_view.main_window import MainWindow

def main():
    app = QApplication(sys.argv)

    color_manager = ColorManager(app)
    color_manager.load_raw("../../../style.qss")

    window = MainWindow(color_manager)

    window.showMaximized()

    sys.exit(app.exec())

if __name__ == '__main__':
    main()
