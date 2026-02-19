from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton, QComboBox, QSpinBox, QWidget
from PySide6.QtGui import QIcon

from nn_verification_visualisation.controller.input_manager.plot_view_controller import PlotViewController
from nn_verification_visualisation.model.data.diagram_config import DiagramConfig
from nn_verification_visualisation.view.base_view.insert_view import InsertView
from nn_verification_visualisation.view.base_view.tutorial_speech_bubble import TutorialSpeechBubble
from nn_verification_visualisation.view.dialogs.settings_dialog import SettingsDialog
from nn_verification_visualisation.view.dialogs.settings_option import SettingsOption
from nn_verification_visualisation.view.plot_view.comparison_loading_widget import ComparisonLoadingWidget
from nn_verification_visualisation.view.plot_view.plot_page import PlotPage
from nn_verification_visualisation.model.data.storage import Storage


class PlotView(InsertView):
    controller: PlotViewController

    settings_remover: Callable[[], None] | None

    def __init__(self, change_view: Callable[[], None], parent=None):
        super().__init__(parent)
        self.settings_remover = None
        self.controller = PlotViewController(self)
        # restore saved diagram tabs
        for diagram in Storage().diagrams:
            self.add_plot_tab(diagram)

        add_button = self._create_simple_icon_button(self.controller.open_plot_generation_dialog,
                                                     ":assets/icons/add_icon.svg")

        view_toggle_button = QPushButton()
        view_toggle_button.clicked.connect(change_view)
        view_toggle_button.setObjectName("switch-button")
        view_toggle_button.setIcon(QIcon(":assets/icons/plot/switch.svg"))

        self.set_bar_corner_widgets([add_button, view_toggle_button], Qt.Corner.TopRightCorner, width=110)

    def add_plot_tab(self, diagram_config: DiagramConfig, index: int = -1):
        '''
        Adds a plot tab to the QTabWidget. Only updates UI, not the backend.
        :param index: position of the new tab (-1 if last)
        :param diagram_config: data of the new tab
        '''
        self.tabs.add_tab(PlotPage(self.controller, diagram_config), index=index)

    def add_loading_tab(self, widget: ComparisonLoadingWidget):
        self.tabs.add_tab(widget)

    def close_tab(self, index: int):
        # If the tab widget has diagram_config (PlotPage), remove it from Storage().diagrams
        w = self.tabs.widget(index)
        diagram = getattr(w, "diagram_config", None)
        if diagram is not None:
            storage = Storage()
            if diagram in storage.diagrams:
                storage.diagrams.remove(diagram)
                storage.request_autosave()
        super().close_tab(index)


    def reload_from_storage(self):
        self.tabs.reset()
        for diagram in Storage().diagrams:
            self.add_plot_tab(diagram)

    def get_default_tab(self) -> QWidget | None:
        text = "1. Use the edit icon in the top right corner to add new comparisons.\n\n2. Select neuron triples with networks, algorithms and specific nodes.\n\n3. Let the algorithms calculate the output bounds from the given selections.\n\n4. View the results and compare them."
        return TutorialSpeechBubble("Quick Tutorial", text)

    def showEvent(self, event, /):
        super().showEvent(event)
        self.settings_remover = SettingsDialog.add_setting(
            SettingsOption("Numer of Directions", self.get_num_directions_changer, "Plot View"))

    def hideEvent(self, event, /):
        super().hideEvent(event)
        if self.settings_remover:
            self.settings_remover()
            self.settings_remover = None

    def get_num_directions_changer(self) -> QWidget:
        def on_change(value):
            Storage().num_directions = value

        changer = QSpinBox()
        changer.setRange(0, 10000)
        changer.setValue(Storage().num_directions)
        changer.valueChanged.connect(on_change)
        return changer
