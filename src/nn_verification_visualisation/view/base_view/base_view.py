from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QStackedWidget, QStackedLayout

from view.base_view.insert_view import InsertView
from view.network_view.network_view import NetworkView
from view.plot_view.plot_view import PlotView


class BaseView(QWidget):
    active_view: InsertView
    plot_view: PlotView
    network_view: NetworkView

    def __init__(self):
        super().__init__()
        self.plot_view = PlotView()
        self.network_view = NetworkView()
        self.active_view = self.network_view
        self.stack = QStackedLayout()
        self.stack.addWidget(self.network_view)
        self.stack.addWidget(self.plot_view)

        container = QWidget()
        container.setLayout(self.stack)

        change_button = QPushButton("Change")
        change_button.clicked.connect(self.change_active_view)

        self.box_layout = QVBoxLayout()
        self.box_layout.setContentsMargins(0,0,0,0)
        self.setLayout(self.box_layout)
        self.layout().addWidget(change_button)
        self.layout().addWidget(container)

    def change_active_view(self):
        if self.active_view is self.network_view:
            index = 1
            self.active_view = self.plot_view
        else:
            index = 0
            self.active_view = self.network_view

        self.stack.setCurrentIndex(index)