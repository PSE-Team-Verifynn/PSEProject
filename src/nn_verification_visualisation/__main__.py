import sys
from pathlib import Path

from nn_verification_visualisation import resources_rc

from PySide6.QtWidgets import QApplication

from nn_verification_visualisation.view.base_view.color_manager import ColorManager
from nn_verification_visualisation.view.base_view.main_window import MainWindow

from nn_verification_visualisation.model.data.storage import Storage

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    color_manager = ColorManager(app)
    style_path = Path(__file__).resolve().parent / "style.qss"
    color_manager.load_raw(str(style_path))

    storage = Storage()
    state_path = Path(__file__).resolve().parent / "save_state.json"
    storage.set_save_state_path(str(state_path))
    storage.load_from_disk()

    window = MainWindow(color_manager)

    window.showMaximized()

    app.aboutToQuit.connect(lambda: Storage().save_to_disk())

    sys.exit(app.exec())

if __name__ == '__main__':
    main()
