from __future__ import annotations
from typing import List, Callable, TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea, QPushButton, QHBoxLayout, QApplication

if TYPE_CHECKING:
    from nn_verification_visualisation.controller.input_manager.plot_view_controller import PlotViewController

from nn_verification_visualisation.model.data.diagram_config import DiagramConfig
from nn_verification_visualisation.utils.result import Result
from nn_verification_visualisation.view.base_view.tab import Tab
from nn_verification_visualisation.view.dialogs.info_popup import InfoPopup
from nn_verification_visualisation.view.dialogs.info_type import InfoType
from nn_verification_visualisation.view.plot_view.pair_loading_widget import PairLoadingWidget
from nn_verification_visualisation.view.plot_view.status import Status

class ComparisonLoadingWidget(Tab):
    diagram_config: DiagramConfig

    __loaders: List[PairLoadingWidget]
    __page_title: QWidget
    __terminate_process: Callable[[int], bool]

    __create_diagram_button: QPushButton

    __controller: PlotViewController

    on_update = Signal(tuple)

    def __init__(self, diagram_config: DiagramConfig, controller: PlotViewController, terminate_process: Callable[[int], bool]):
        self.diagram_config = diagram_config
        self.__terminate_process = terminate_process
        self.__controller = controller

        super().__init__(f"Loading {diagram_config.get_title()}", ":assets/icons/plot/hourglass.svg", has_sidebar=False, remove_close_button=True)

        self.on_update.connect(lambda x: self.loading_updated(x[0], x[1]))


    def get_content(self) -> QWidget:
        content = QWidget()
        self.__page_title = QLabel("Loading...")
        self.__page_title.setObjectName("title")

        self.__loaders = self.__get_pair_list()

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        pair_container = QWidget()

        pair_layout = QVBoxLayout(pair_container)

        pair_layout.addStretch()

        for loader in self.__loaders:
            pair_layout.addWidget(loader, alignment=Qt.AlignmentFlag.AlignHCenter)

        self.__create_diagram_button = QPushButton("Continue to Diagram Tab")
        self.__create_diagram_button.setVisible(False)
        self.__create_diagram_button.clicked.connect(self.__create_diagram_tab)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.__create_diagram_button)
        button_layout.addStretch()

        pair_layout.addLayout(button_layout)

        pair_layout.addStretch()
        scroll_area.setWidget(pair_container)

        layout = QVBoxLayout()
        layout.addWidget(self.__page_title)
        layout.addWidget(scroll_area)

        content.setLayout(layout)
        return content

    def __create_diagram_tab(self):
        self.__controller.create_diagram_tab(self)

    def loading_updated(self, index: int, result: Result):
        QApplication.processEvents()
        loader = self.__loaders[index]
        loader.set_status(Status.Done if result.is_success else Status.Failed)
        if not result.is_success:
            loader.error = result.error
        QApplication.processEvents()

    def loading_finished(self):
        self.__create_diagram_button.setVisible(True)
        pass

    def __on_clicked(self, index: int) -> None:
        loader = self.__loaders[index]
        if loader.status == Status.Ongoing: # on waiting -> cancel
            self.__terminate_process(index)
        elif loader.status == Status.Failed and loader.error is not None: # on error -> show error
            error_message = str(loader.error)
            error_dialog = InfoPopup(self.__controller.current_plot_view.close_dialog, error_message, InfoType.ERROR)
            self.__controller.current_plot_view.open_dialog(error_dialog)
        pass

    def __get_pair_list(self) -> List[PairLoadingWidget]:
        loaders: List[PairLoadingWidget] = []
        for i, config in enumerate(self.diagram_config.plot_generation_configs):
            loader = PairLoadingWidget(config.get_title(), lambda : self.__on_clicked(i))
            loader.set_status(Status.Ongoing)
            loaders.append(loader)
        return loaders
