import sys
from pathlib import Path

from nn_verification_visualisation import resources_rc

from PySide6.QtWidgets import QApplication

from nn_verification_visualisation.view.base_view.color_manager import ColorManager
from nn_verification_visualisation.view.base_view.main_window import MainWindow

from nn_verification_visualisation.model.data.storage import Storage
from nn_verification_visualisation.model.data_loader.save_state_loader import SaveStateLoader
from nn_verification_visualisation.view.dialogs.info_popup import InfoPopup
from nn_verification_visualisation.view.dialogs.info_type import InfoType

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    color_manager = ColorManager(app)
    style_path = Path(__file__).resolve().parent / "style.qss"
    color_manager.load_raw(str(style_path))

    storage = Storage()
    state_path = Path(__file__).resolve().parent / "save_state.json"
    storage.set_save_state_path(str(state_path))
    has_existing_state_file = state_path.exists()
    load_res = storage.load_from_disk()

    window = MainWindow(color_manager)

    window.showMaximized()

    if has_existing_state_file and not load_res.is_success:
        text = f"Could not load saved project:\n{load_res.error}"
        dialog = InfoPopup(window.base_view.active_view.close_dialog, text, InfoType.WARNING)
        window.base_view.active_view.open_dialog(dialog)
    elif has_existing_state_file:
        warnings = SaveStateLoader().get_warnings()
        if warnings:
            message = "Loaded saved project with warnings:\n" + "\n".join(f"- {w}" for w in warnings)
            dialog = InfoPopup(window.base_view.active_view.close_dialog, message, InfoType.WARNING)
            window.base_view.active_view.open_dialog(dialog)

    app.aboutToQuit.connect(lambda: Storage().save_to_disk())

    sys.exit(app.exec())

if __name__ == '__main__':
    main()
