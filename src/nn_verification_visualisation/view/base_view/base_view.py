import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QStyleHints
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QStackedWidget, QStackedLayout, QHBoxLayout, QComboBox

from nn_verification_visualisation.view.base_view.color_manager import ColorManager
from nn_verification_visualisation.view.base_view.insert_view import InsertView
from nn_verification_visualisation.view.dialogs.settings_dialog import SettingsDialog
from nn_verification_visualisation.view.dialogs.settings_option import SettingsOption
from nn_verification_visualisation.view.network_view.network_view import NetworkView
from nn_verification_visualisation.view.plot_view.plot_view import PlotView


class BaseView(QWidget):
    active_view: InsertView
    plot_view: PlotView
    network_view: NetworkView
    color_manager: ColorManager

    contrast_preference: Qt.ContrastPreference

    def __init__(self, color_manager: ColorManager, parent=None):
        super().__init__(parent)
        self.color_manager = color_manager
        self.color_manager.set_colors(ColorManager.NETWORK_COLORS)

        self.plot_view = PlotView(self.change_active_view, parent=self)
        self.network_view = NetworkView(self.change_active_view, parent=self)
        self.active_view = self.network_view
        self.stack = QStackedLayout()
        self.stack.addWidget(self.network_view)
        self.stack.addWidget(self.plot_view)

        # this is done to prevent C++ object deletion
        self.style_hints = self.color_manager.app.styleHints()
        self.accessibility_hints = self.style_hints.accessibility()
        self.contrast_preference = self.accessibility_hints.contrastPreference()

        container = QWidget()
        container.setLayout(self.stack)

        self.box_layout = QVBoxLayout()
        self.box_layout.setContentsMargins(0, 0, 0, 0)
        self.box_layout.setSpacing(0)
        self.box_layout.addWidget(container)
        self.setLayout(self.box_layout)

        SettingsDialog.add_setting(SettingsOption("Color Scheme", self.get_color_mode_changer, "Appearance"))
        SettingsDialog.add_setting(SettingsOption("High Contrast", self.get_high_contrast_changer, "Appearance"))

    def change_active_view(self):
        t1 = time.time()
        old_view = self.active_view
        old_view.setUpdatesEnabled(False)
        if self.active_view is self.network_view:
            index = 1
            self.active_view = self.plot_view
            new_colors = ColorManager.DIAGRAM_COLORS
        else:
            index = 0
            self.active_view = self.network_view
            new_colors= ColorManager.NETWORK_COLORS

        self.stack.setCurrentIndex(index)
        print(f"Stack switch: {(time.time() - t1) * 1000:.0f}ms")
        t2 = time.time()
        self.color_manager.set_colors(new_colors)
        print(f"Color change: {(time.time() - t2) * 1000:.0f}ms")  # ← This is slow!
        self.active_view.setUpdatesEnabled(True)

    def get_color_mode_changer(self):
        change_widget = QComboBox()
        change_widget.addItems(["Light", "Dark"])
        color_scheme = self.color_manager.app.styleHints().colorScheme()
        match color_scheme:
            case Qt.ColorScheme.Dark:
                change_widget.setCurrentIndex(1)
            case Qt.ColorScheme.Light:
                change_widget.setCurrentIndex(0)
            case _:
                change_widget.setCurrentIndex(1)

        change_widget.currentIndexChanged.connect(self.change_color_scheme)
        return change_widget


    def change_color_scheme(self, index: int):
        color_scheme = Qt.ColorScheme.Dark
        match index:
            case 0:
                color_scheme = Qt.ColorScheme.Light
            case 1:
                color_scheme = Qt.ColorScheme.Dark

        self.color_manager.app.styleHints().setColorScheme(color_scheme)
        self.color_manager.update_colors()

    def get_high_contrast_changer(self):
        change_widget = QComboBox()
        change_widget.addItems(["Off", "On"])

        use_safed_preference = self.contrast_preference == self.accessibility_hints.contrastPreference()

        is_high_contrast = self.contrast_preference if use_safed_preference else self.accessibility_hints.contrastPreference()
        if not self.contrast_preference:
            self.contrast_preference = is_high_contrast

        match is_high_contrast:
            case Qt.ContrastPreference.NoPreference:
                change_widget.setCurrentIndex(0)
            case Qt.ContrastPreference.HighContrast:
                change_widget.setCurrentIndex(1)


        change_widget.currentIndexChanged.connect(self.change_contrast_mode)
        return change_widget


    def change_contrast_mode(self, index: int):
        contrast_mode = Qt.ContrastPreference.NoPreference
        match index:
            case 0:
                contrast_mode = Qt.ContrastPreference.NoPreference
            case 1:
                contrast_mode = Qt.ContrastPreference.HighContrast

        self.contrast_preference = contrast_mode
        self.color_manager.update_accessibility(contrast_mode)

