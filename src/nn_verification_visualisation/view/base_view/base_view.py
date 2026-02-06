from PySide6.QtWidgets import QWidget, QVBoxLayout, QStackedLayout

from nn_verification_visualisation.view.base_view.color_manager import ColorManager
from nn_verification_visualisation.view.base_view.insert_view import InsertView
from nn_verification_visualisation.view.network_view.network_view import NetworkView
from nn_verification_visualisation.view.plot_view.plot_view import PlotView


class BaseView(QWidget):
    active_view: InsertView
    plot_view: PlotView
    network_view: NetworkView
    color_manager: ColorManager

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

        container = QWidget()
        container.setLayout(self.stack)

        self.box_layout = QVBoxLayout()
        self.box_layout.setContentsMargins(0, 0, 0, 0)
        self.box_layout.setSpacing(0)
        self.box_layout.addWidget(container)
        self.setLayout(self.box_layout)

    def change_active_view(self):
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
        self.color_manager.set_colors(new_colors)
        self.active_view.setUpdatesEnabled(True)

    def reload_from_storage(self):
        self.network_view.reload_from_storage()
        self.plot_view.reload_from_storage()
