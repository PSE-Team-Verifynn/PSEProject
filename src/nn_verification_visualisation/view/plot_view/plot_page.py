from typing import List, Callable

from PySide6.QtCore import Qt, QEvent
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)

from nn_verification_visualisation.model.data.diagram_config import DiagramConfig
from nn_verification_visualisation.controller.input_manager.plot_view_controller import PlotViewController
from nn_verification_visualisation.view.base_view.plot_settings_widget import PlotSettingsWidget
from nn_verification_visualisation.view.plot_view.plot_widget import PlotWidget
from nn_verification_visualisation.view.base_view.tab import Tab
from nn_verification_visualisation.view.dialogs.settings_dialog import SettingsDialog
from nn_verification_visualisation.view.dialogs.settings_option import SettingsOption
from nn_verification_visualisation.view.dialogs.neuron_picker import get_neuron_colors

class PlotPage(Tab):
    plot_widgets: list[PlotWidget]
    plot_setting_widgets: list[PlotSettingsWidget]

    setting_remover: Callable[[], None] | None

    controller: PlotViewController
    diagram_config: DiagramConfig
    locked: List[PlotWidget]
    __plot_grid: QGridLayout
    __syncing: bool
    __scroll_area: QScrollArea | None
    __grid_host: QWidget | None
    __bottom_spacer_height: int
    __plots_sidebar_layout: QVBoxLayout
    __node_pairs_list: QListWidget | None
    __node_pairs_layout: QVBoxLayout | None

    def __init__(self, controller: PlotViewController, diagram_config: DiagramConfig):
        self.diagram_config = diagram_config
        self.__syncing = False
        self.__scroll_area = None
        self.__grid_host = None
        self.__bottom_spacer_height = 32
        self.controller = controller

        self.setting_remover = None

        self.plot_widgets = []
        self.plot_setting_widgets = []

        super().__init__("Example Tab", ":assets/icons/plot/chart.svg")

        # add start plots
        for i in range(len(self.diagram_config.polygons)):
            self.__add_plot([i])

        # configuration is currently not implemented
        # self.configuration = configuration

    def __update_selection(self, widget: PlotSettingsWidget, sel: list[int]):
        widget.set_selection(sel)

        # update PlotWidget
        length = len(self.diagram_config.polygons)
        colors = get_neuron_colors(length)
        index = self.plot_setting_widgets.index(widget)
        self.plot_widgets[index].render_plot([self.diagram_config.polygons[i] for i in sel], [colors[i] for i in sel],
                                             [f"Pair {i + 1}" for i in sel])
    def __delete_plot(self, widget: PlotSettingsWidget):
        index = self.plot_setting_widgets.index(widget)

        # remove from DiagramConfig
        self.diagram_config.plots.pop(index)

        # remove from side panel
        self.plot_setting_widgets.remove(widget)
        widget.setParent(None)

        #remove from main panel
        plot_widget = self.plot_widgets.pop(index)
        plot_widget.setParent(None)

        self.__relayout_plots()


    def __add_plot(self, plot: list[int]):
        # add diagram to DiagramConfig
        self.diagram_config.plots.append(plot)
        index = len(self.diagram_config.plots)

        title_text = f"Diagram {index}"

        # add sidebar panel
        settings_widget = PlotSettingsWidget(title_text, self.diagram_config, self.__update_selection, self.__delete_plot)
        self.plot_setting_widgets.append(settings_widget)
        self.__plots_sidebar_layout.addWidget(settings_widget)

        # add main panel
        plot_widget = PlotWidget(title=title_text, on_limits_changed=self.__on_limits_changed)
        plot_widget.setFixedSize(self.controller.card_size, self.controller.card_size)

        self.plot_widgets.append(plot_widget)
        self.__plot_grid.addWidget(plot_widget)

        self.__update_selection(settings_widget, plot)
        self.__relayout_plots()

    def get_content(self) -> QWidget:
        container = QWidget()

        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        self.__scroll_area = scroll_area

        grid_host = QWidget()
        grid_host.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        self.__plot_grid = QGridLayout(grid_host)
        self.__plot_grid.setContentsMargins(4, 4, 4, 4)
        self.__plot_grid.setSpacing(12)
        self.__plot_grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.__grid_host = grid_host

        scroll_area.setWidget(grid_host)
        layout.addWidget(scroll_area)

        scroll_area.viewport().installEventFilter(self)
        self.__relayout_plots()
        return container
        # return ComparisonLoadingWidget()

    def get_side_bar(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        title = QLabel("Settings")
        title.setObjectName("title")
        layout.addWidget(title)

        self.__plots_sidebar_layout = QVBoxLayout()
        self.__plots_sidebar_layout.setContentsMargins(0, 0, 0, 0)
        self.__plots_sidebar_layout.setSpacing(8)
        layout.addLayout(self.__plots_sidebar_layout)

        add_diagram_button = QPushButton("Add Diagram")
        add_diagram_button.clicked.connect(lambda: self.__add_plot([0]))
        layout.addWidget(add_diagram_button, alignment=Qt.AlignmentFlag.AlignLeft)

        layout.addStretch(1)

        return container

    def __relayout_plots(self):
        if self.__scroll_area is None:
            return
        viewport_width = self.__scroll_area.viewport().width()
        margins = self.__plot_grid.contentsMargins()
        spacing = self.__plot_grid.horizontalSpacing()
        available_width = viewport_width - margins.left() - margins.right()
        if available_width <= 0:
            return
        if spacing < 0:
            spacing = 0
        columns = max(1, (available_width + spacing) // (self.controller.card_size + spacing))
        required_width = (
                columns * self.controller.card_size
                + max(0, columns - 1) * spacing
                + margins.left()
                + margins.right()
        )

        while self.__plot_grid.count():
            item = self.__plot_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(self.__grid_host)

        if self.__grid_host is not None:
            self.__grid_host.setFixedWidth(required_width)

        for index, widget in enumerate(self.plot_widgets):
            row = index // columns
            col = index % columns
            self.__plot_grid.addWidget(widget, row, col)

        spacer_row = (len(self.plot_widgets) + columns - 1) // columns
        spacer = QSpacerItem(0, self.__bottom_spacer_height, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.__plot_grid.addItem(spacer, spacer_row, 0, 1, columns)

    def __on_card_size_changed(self, value: int):
        self.controller.set_card_size(value)
        for widget in self.plot_widgets:
            widget.setFixedSize(self.controller.card_size, self.controller.card_size)
        self.__relayout_plots()

    def get_card_size_changer(self) -> QWidget:
        size_slider = QSlider(Qt.Orientation.Horizontal)
        size_slider.setMinimum(320)
        size_slider.setMaximum(560)
        size_slider.setValue(self.controller.card_size)
        size_slider.setSingleStep(10)
        size_slider.valueChanged.connect(self.__on_card_size_changed)
        return size_slider

    def __on_limits_changed(self, source: PlotWidget):
        if self.__syncing or not getattr(source, "locked", False):
            return
        ax = source.axes
        if ax is None:
            return
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        self.__syncing = True
        for widget in self.plot_widgets:
            if widget is source or not getattr(widget, "locked", False):
                continue
            target_ax = widget.axes
            target_canvas = widget.canvas
            if target_ax is None or target_canvas is None:
                continue
            target_ax.set_xlim(xlim)
            target_ax.set_ylim(ylim)
            target_canvas.draw_idle()
        self.__syncing = False

    def eventFilter(self, watched, event):
        if watched is self.__scroll_area.viewport() and event.type() == QEvent.Type.Resize:
            self.__relayout_plots()
        return super().eventFilter(watched, event)

    def showEvent(self, event):
        self.setting_remover = SettingsDialog.add_setting(
            SettingsOption("Plot Card Size", self.get_card_size_changer, "Plot View"))

    def hideEvent(self, event, /):
        if self.setting_remover:
            self.setting_remover()
            self.setting_remover = None