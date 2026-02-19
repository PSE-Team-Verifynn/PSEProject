from __future__ import annotations
from typing import List, Callable, TYPE_CHECKING

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
if TYPE_CHECKING:
    from nn_verification_visualisation.controller.input_manager.plot_view_controller import PlotViewController

from nn_verification_visualisation.model.data.diagram_config import DiagramConfig
from nn_verification_visualisation.view.base_view.plot_settings_widget import PlotSettingsWidget
from nn_verification_visualisation.view.plot_view.plot_widget import PlotWidget
from nn_verification_visualisation.view.base_view.tab import Tab
from nn_verification_visualisation.view.dialogs.settings_dialog import SettingsDialog
from nn_verification_visualisation.view.dialogs.settings_option import SettingsOption
from nn_verification_visualisation.view.dialogs.neuron_picker import get_neuron_colors
from nn_verification_visualisation.model.data.storage import Storage


def _polygon_is_3d(polygon: list) -> bool:
    """
    Returns True when *polygon* is 3-D face data (list of triangular faces),
    False when it is a flat 2-D vertex list.

    2-D: list[tuple[float, float]]           — first element is a 2-float tuple
    3-D: list[list[tuple[float,float,float]]] — first element is itself a list of 3-tuples
    """
    if not polygon:
        return False
    first = polygon[0]
    # In 3-D, each entry is a face (a sequence of vertices), so its own first
    # element is a vertex tuple.  In 2-D, each entry is already a vertex tuple.
    return isinstance(first, (list, tuple)) and bool(first) and isinstance(first[0], (list, tuple))


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

        super().__init__(diagram_config.get_title(), ":assets/icons/plot/chart.svg")

        # Build plots from saved state if present, otherwise one per polygon
        self.__initializing = True

        if not getattr(self.diagram_config, "plots", None):
            self.diagram_config.plots = [
                [i] for i in range(len(self.diagram_config.polygons))
            ]
            Storage().request_autosave()

        for sel in self.diagram_config.plots:
            self.__add_plot(list(sel), update_config=False)

        self.__initializing = False

    def __update_selection(self, widget: PlotSettingsWidget, sel: list[int]):
        index = self.plot_setting_widgets.index(widget)

        if 0 <= index < len(self.diagram_config.plots):
            self.diagram_config.plots[index] = list(sel)
            if not getattr(self, "_PlotPage__initializing", False):
                Storage().request_autosave()

        widget.set_selection(sel)

        length = len(self.diagram_config.polygons)
        colors = get_neuron_colors(length)

        # Determine required dimensionality from the first non-empty selected
        # polygon. All pairs in a diagram should share the same neuron count,
        # but a user can check a mix; we treat the first selected as canonical.
        is_3d = False
        for i in sel:
            if self.diagram_config.polygons[i]:
                is_3d = _polygon_is_3d(self.diagram_config.polygons[i])
                break

        # Filter to only polygons that match the detected dimensionality so we
        # never pass a mixed 2D/3D list into a single render_plot call.
        compatible_sel = [
            i for i in sel
            if not self.diagram_config.polygons[i]          # empty — harmless
            or _polygon_is_3d(self.diagram_config.polygons[i]) == is_3d
        ]

        current_widget = self.plot_widgets[index]
        current_is_3d = current_widget.__class__.__name__ == "PlotWidget3D"

        # Swap the widget subclass if the dimensionality has changed.
        if is_3d != current_is_3d:
            new_widget = PlotWidget.make_plot_widget(
                on_limits_changed=self.__on_limits_changed,
                title=current_widget.title,
                is_3d=is_3d,
            )
            new_widget.setFixedSize(self.controller.card_size, self.controller.card_size)
            new_widget.locked = current_widget.locked
            current_widget.setParent(None)
            self.plot_widgets[index] = new_widget
            current_widget = new_widget

        current_widget.render_plot(
            [self.diagram_config.polygons[i] for i in compatible_sel],
            [colors[i] for i in compatible_sel],
            [f"Pair {i + 1}" for i in compatible_sel],
        )

        # Re-insert swapped widget into the grid at the correct position.
        if is_3d != current_is_3d:
            self.__relayout_plots()

    def __delete_plot(self, widget: PlotSettingsWidget):
        index = self.plot_setting_widgets.index(widget)

        self.diagram_config.plots.pop(index)
        Storage().request_autosave()

        self.plot_setting_widgets.remove(widget)
        widget.setParent(None)

        plot_widget = self.plot_widgets.pop(index)
        plot_widget.setParent(None)

        self.__relayout_plots()

    def __add_plot(self, plot: list[int], update_config: bool = True):
        if update_config:
            self.diagram_config.plots.append(list(plot))
            Storage().request_autosave()

        index = len(self.plot_setting_widgets) + 1
        title_text = f"Diagram {index}"

        # Sidebar panel
        settings_widget = PlotSettingsWidget(
            title_text, self.diagram_config,
            self.__update_selection, self.__delete_plot,
        )
        self.plot_setting_widgets.append(settings_widget)
        self.__plots_sidebar_layout.addWidget(settings_widget)

        # Create a placeholder widget but do NOT add it to the grid yet.
        # __update_selection may immediately swap it for a 3-D widget; if the
        # placeholder were already in the grid, setParent(None) during the swap
        # would detach it from the layout and Qt would show it as a floating
        # top-level window.
        plot_widget = PlotWidget.make_plot_widget(
            on_limits_changed=self.__on_limits_changed,
            title=title_text,
            is_3d=False,
        )
        plot_widget.setFixedSize(self.controller.card_size, self.controller.card_size)
        self.plot_widgets.append(plot_widget)

        # Resolve the correct widget type (may replace plot_widget in self.plot_widgets).
        # __update_selection calls __relayout_plots internally when a swap occurs,
        # which already adds the final widget to the grid via __relayout_plots.
        # If no swap occurs, we add and relayout here.
        self.__update_selection(settings_widget, plot)
        final_widget = self.plot_widgets[-1]
        if final_widget is plot_widget:
            # No swap happened — widget was never added to the grid yet.
            self.__plot_grid.addWidget(final_widget)
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

        layout.addStretch(1)

        add_diagram_button = QPushButton("Add Diagram")
        add_diagram_button.clicked.connect(lambda: self.__add_plot([0]))
        add_diagram_button.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(add_diagram_button, alignment=Qt.AlignmentFlag.AlignBottom)

        edit_nodes_button = QPushButton("Edit Nodes")
        edit_nodes_button.clicked.connect(lambda: self.__start_edit())
        edit_nodes_button.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(edit_nodes_button, alignment=Qt.AlignmentFlag.AlignBottom)

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
        columns = max(
            1, (available_width + spacing) // (self.controller.card_size + spacing)
        )
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
        spacer = QSpacerItem(
            0, self.__bottom_spacer_height,
            QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed,
        )
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
        """
        Propagates the view state from *source* to all other locked peers.
        Works for both 2-D (xlim/ylim) and 3-D (elev/azim/xlim/ylim/zlim)
        because each subclass implements get_view_state / apply_view_state.
        """
        if self.__syncing or not getattr(source, "locked", False):
            return

        state = source.get_view_state()

        self.__syncing = True
        try:
            for widget in self.plot_widgets:
                if widget is source or not getattr(widget, "locked", False):
                    continue
                widget.apply_view_state(state)
        finally:
            self.__syncing = False

    def __start_edit(self):
        self.controller.open_plot_generation_editing_dialog(
            self.diagram_config.plot_generation_configs, self
        )

    def eventFilter(self, watched, event):
        if (
            self.__scroll_area is not None
            and watched is self.__scroll_area.viewport()
            and event.type() == QEvent.Type.Resize
        ):
            self.__relayout_plots()
        return super().eventFilter(watched, event)

    def showEvent(self, event):
        super().showEvent(event)
        self.setting_remover = SettingsDialog.add_setting(
            SettingsOption("Plot Card Size", self.get_card_size_changer, "Plot View")
        )

    def hideEvent(self, event, /):
        super().hideEvent(event)
        if self.setting_remover:
            self.setting_remover()
            self.setting_remover = None