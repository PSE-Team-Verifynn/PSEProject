from __future__ import annotations
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QWidget, QMenu, QGraphicsDropShadowEffect, QApplication, QMainWindow, QFileDialog

from nn_verification_visualisation.view.dialogs.info_popup import InfoPopup
from nn_verification_visualisation.view.dialogs.info_type import InfoType

from typing import TYPE_CHECKING

from nn_verification_visualisation.view.dialogs.settings_dialog import SettingsDialog
from nn_verification_visualisation.model.data.storage import Storage
from nn_verification_visualisation.model.data_loader.save_state_loader import SaveStateLoader
from nn_verification_visualisation.model.data_exporter.save_state_exporter import SaveStateExporter

if TYPE_CHECKING:
    from nn_verification_visualisation.view.base_view.insert_view import InsertView
    from nn_verification_visualisation.view.base_view.base_view import BaseView


class ActionMenu(QWidget):

    def __init__(self, parent: InsertView):
        super().__init__()
        self.menu = QMenu()
        self.parent = parent
        self.menu.setWindowFlags(
            Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint | Qt.WindowType.NoDropShadowWindowHint
        )
        self.menu.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        settings_action = self.menu.addAction("Settings")
        settings_action.triggered.connect(self.__settings_action)

        open_project_action = self.menu.addAction("Open Project")
        open_project_action.triggered.connect(self.__open_project_action)

        export_project_action = self.menu.addAction("Export Project")
        export_project_action.triggered.connect(self.__export_project_action)

        exit_action = self.menu.addAction("Exit")
        exit_action.triggered.connect(self.__exit_action)

        shadow = QGraphicsDropShadowEffect(self.menu)
        shadow.setBlurRadius(10)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(0, 0, 0, 60))

        self.menu.setGraphicsEffect(shadow)

    def __settings_action(self):
        settings_dialog = SettingsDialog(self.parent.close_dialog)
        self.parent.open_dialog(settings_dialog)

    def __open_project_action(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self.parent,
            "Open Project",
            ".",
            "Project Files (*.json);;All Files (*)",
        )
        if file_path == "":
            return

        result = SaveStateLoader().load_save_state(file_path)
        if not result.is_success:
            self.parent.open_dialog(InfoPopup(
                self.parent.close_dialog,
                f"Could not open project:\n{result.error}",
                InfoType.ERROR
            ))
            return

        storage = Storage()
        storage.load_save_state(result.data)
        storage.set_save_state_path(file_path)

        base_view = self.__find_base_view()
        if base_view is not None:
            base_view.reload_from_storage()

        self.parent.open_dialog(InfoPopup(
            self.parent.close_dialog,
            "Project opened successfully.",
            InfoType.CONFIRMATION
        ))

    def __export_project_action(self):
        default_name = "project_export.json"
        file_path, _ = QFileDialog.getSaveFileName(
            self.parent,
            "Export Project",
            str(Path.cwd() / default_name),
            "Project Files (*.json);;All Files (*)",
        )
        if file_path == "":
            return

        path = Path(file_path)
        if path.suffix.lower() != ".json":
            path = path.with_suffix(".json")

        export_res = SaveStateExporter().export_save_state(Storage().get_save_state())
        if not export_res.is_success:
            self.parent.open_dialog(InfoPopup(
                self.parent.close_dialog,
                f"Could not export project:\n{export_res.error}",
                InfoType.ERROR
            ))
            return

        try:
            path.write_text(export_res.data, encoding="utf-8")
        except BaseException as e:
            self.parent.open_dialog(InfoPopup(
                self.parent.close_dialog,
                f"Could not export project:\n{e}",
                InfoType.ERROR
            ))
            return

        self.parent.open_dialog(InfoPopup(
            self.parent.close_dialog,
            f"Project exported to:\n{path}",
            InfoType.CONFIRMATION
        ))

    def __find_base_view(self) -> BaseView | None:
        current = self.parent
        while current is not None:
            if current.__class__.__name__ == "BaseView":
                return current
            current = current.parent()
        return None

    def __exit_action(self):
        """
        Closes the main window of the application
        """
        app = QApplication.instance()
        main_window = None

        # Find the MainWindow instance
        for widget in app.topLevelWidgets():
            if isinstance(widget, QMainWindow):
                main_window = widget
                break

        if main_window:
            Storage().save_to_disk()
            main_window.close()
