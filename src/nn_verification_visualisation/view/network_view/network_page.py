from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QPushButton, QHBoxLayout, QDoubleSpinBox, QDataWidgetMapper, \
    QListWidget, QListWidgetItem, QGroupBox

from nn_verification_visualisation.controller.input_manager.network_view_controller import NetworkViewController
from nn_verification_visualisation.model.data.network_verification_config import NetworkVerificationConfig
from nn_verification_visualisation.view.base_view.tab import Tab
from nn_verification_visualisation.view.network_view.network_widget import NetworkWidget
from nn_verification_visualisation.view.base_view.sample_metrics import SampleMetricsWidget
from nn_verification_visualisation.view.dialogs.sample_results_dialog import SampleResultsDialog
from nn_verification_visualisation.view.base_view.bounds_display import BoundsDisplayWidget


class NetworkPage(Tab):
    configuration: NetworkVerificationConfig
    controller: NetworkViewController
    input_count: int

    def __init__(self, controller: NetworkViewController, configuration: NetworkVerificationConfig):
        self.configuration = configuration
        self.controller = controller
        self.input_count = configuration.layers_dimensions[0]
        super().__init__(configuration.network.name, ":assets/icons/network/network.svg")

    def get_content(self) -> QWidget:
        return NetworkWidget(self.configuration)

    def get_side_bar(self) -> QWidget:
        base = QWidget()

        title = QLabel("Input Layer Bounds")
        title.setObjectName("title")

        bounds_group = QGroupBox("Bounds")
        bounds_group_layout = QVBoxLayout(bounds_group)
        bounds_group_layout.setContentsMargins(6, 6, 6, 6)
        bounds_group_layout.setSpacing(4)

        self.bounds_list = QListWidget()

        row_height = self.bounds_list.sizeHintForRow(0)
        if row_height <= 0:
            row_height = self.bounds_list.fontMetrics().height() + 8
        self.bounds_list.setMinimumHeight(row_height * 5 + self.bounds_list.frameWidth() * 2)
        self.bounds_list.currentRowChanged.connect(self.__on_bounds_selection_changed)
        bounds_group_layout.addWidget(self.bounds_list)
        bounds_group_layout.addSpacing(5)

        remove_button = QPushButton("Remove Selected Bounds")
        remove_button.clicked.connect(self.__on_remove_bounds_clicked)
        bounds_group_layout.addWidget(remove_button)

        self.add_button = QPushButton("Add Bounds")
        self.add_button.setObjectName("background-button")
        self.add_button.clicked.connect(self.__on_add_bounds_clicked)
        bounds_group_layout.addWidget(self.add_button)

        self.run_samples_button = QPushButton("Run Samples")
        self.run_samples_button.setObjectName("background-button")
        self.run_samples_button.clicked.connect(self.__on_run_samples_clicked)
        bounds_group_layout.addWidget(self.run_samples_button)

        save_button = QPushButton("Save Bounds")
        save_button.setObjectName("background-button")
        save_button.clicked.connect(self.__on_save_bounds_clicked)

        import_button = QPushButton("Import Bounds")
        import_button.setObjectName("background-button")
        import_button.clicked.connect(lambda: self.controller.load_bounds(self.configuration))

        bound_input_layout = QVBoxLayout()
        bound_input_layout.setContentsMargins(6, 6, 6, 6)
        bound_input_layout.setSpacing(4)

        model = self.configuration.bounds

        self.mappers = []  # keeps mappers in memory
        self.bound_inputs = []

        for i in range(self.input_count):
            row_layout = QHBoxLayout()

            label = QLabel("{}:".format(i))
            label.setObjectName("label")

            mapper = QDataWidgetMapper()
            mapper.setModel(model)

            min_input = QDoubleSpinBox(minimum=-999999, maximum=999999)
            min_input.setFixedWidth(75)
            min_input.setSingleStep(.1)
            mapper.addMapping(min_input, 0)

            max_input = QDoubleSpinBox(minimum=-999999, maximum=999999)
            max_input.setFixedWidth(75)
            max_input.setSingleStep(.1)
            mapper.addMapping(max_input, 1)

            row_layout.addWidget(label)
            row_layout.addWidget(min_input)
            row_layout.addWidget(max_input)

            mapper.setCurrentIndex(i)
            bound_input_layout.addLayout(row_layout)

            self.mappers.append(mapper)
            self.bound_inputs.append((min_input, max_input))

        layout = QVBoxLayout()
        layout.addWidget(title)
        layout.addWidget(bounds_group)
        self.edit_group = QGroupBox("Edit Bounds")
        edit_group_layout = QVBoxLayout(self.edit_group)
        edit_group_layout.setContentsMargins(6, 6, 6, 6)
        edit_group_layout.setSpacing(6)
        edit_group_layout.addLayout(bound_input_layout)

        actions_layout = QHBoxLayout()
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(6)
        actions_layout.addWidget(import_button)
        actions_layout.addWidget(save_button)
        edit_group_layout.addLayout(actions_layout)

        self.display_group = BoundsDisplayWidget("Bounds", scrollable=False)
        self.display_group.set_rows(self.input_count)

        layout.addWidget(self.edit_group)
        layout.addWidget(self.display_group)
        self.sample_metrics = SampleMetricsWidget("Sample Results", include_min=False, max_items=10, scrollable=False)
        self.sample_metrics.setVisible(False)
        layout.addWidget(self.sample_metrics)
        self.full_results_button = QPushButton("Full Results")
        self.full_results_button.setVisible(False)
        self.full_results_button.setEnabled(False)
        self.full_results_button.clicked.connect(self.__on_full_results_clicked)
        layout.addWidget(self.full_results_button)
        layout.addStretch(1)
        base.setLayout(layout)

        self.__refresh_bounds_list()
        self.__set_edit_mode(False)
        self.__update_sample_results()

        return base

    def on_changed(self, bounds_num: int, is_max: bool, new_val: float):
        self.configuration.bounds.bounds[bounds_num] = (new_val, new_val)
        print("{} changed to {}".format(bounds_num, new_val))

    def __on_save_bounds_clicked(self):
        selected_index = self.controller.save_bounds(self.configuration)
        self.__refresh_bounds_list(selected_index)
        self.__set_bounds_editable(False)
        self.__set_edit_mode(False)
        self.__update_display_bounds()

    def __on_bounds_selection_changed(self, row: int):
        if row < 0:
            self.controller.select_bounds(self.configuration, None)
            self.__set_bounds_editable(True)
            self.__set_edit_mode(True)
            self.__update_sample_results()
            return

        self.controller.select_bounds(self.configuration, row)
        self.__set_bounds_editable(False)
        self.__set_edit_mode(False)
        self.__update_display_bounds()
        self.__update_sample_results()

    def __on_remove_bounds_clicked(self):
        row = self.bounds_list.currentRow()
        if row < 0:
            return
        removed = self.controller.remove_bounds(self.configuration, row)
        if removed:
            new_count = len(self.configuration.saved_bounds)
            next_index = min(row, new_count - 1) if new_count > 0 else None
            self.__refresh_bounds_list(next_index)
            self.__update_display_bounds()
            self.__update_sample_results()

    def __on_add_bounds_clicked(self):
        self.bounds_list.clearSelection()
        self.controller.select_bounds(self.configuration, None)
        self.__set_bounds_editable(True)
        self.__set_edit_mode(True)
        self.__update_samples_action()
        self.__update_sample_results()

    def __on_run_samples_clicked(self):
        self.controller.open_run_samples_dialog(self.configuration, on_results=lambda _res: self.__update_sample_results())

    def __refresh_bounds_list(self, selected_row: int | None = None):
        self.bounds_list.blockSignals(True)
        self.bounds_list.clear()
        for i, _bounds in enumerate(self.configuration.saved_bounds):
            self.bounds_list.addItem(QListWidgetItem(f"Bounds {i + 1:02d}"))
        if selected_row is None:
            selected_row = self.configuration.selected_bounds_index
        if selected_row is not None and 0 <= selected_row < self.bounds_list.count():
            self.bounds_list.setCurrentRow(selected_row)
            self.controller.select_bounds(self.configuration, selected_row)
            self.__set_bounds_editable(False)
            self.__set_edit_mode(False)
            self.__update_display_bounds()
        else:
            self.bounds_list.clearSelection()
            self.controller.select_bounds(self.configuration, None)
            self.__set_bounds_editable(True)
            self.__set_edit_mode(False)
            self.__update_display_bounds()
        self.__update_samples_action()
        self.__update_sample_results()
        self.bounds_list.blockSignals(False)

    def __set_bounds_editable(self, editable: bool):
        for min_input, max_input in self.bound_inputs:
            min_input.setEnabled(editable)
            max_input.setEnabled(editable)

    def __set_edit_mode(self, active: bool):
        has_bounds = len(self.configuration.saved_bounds) > 0
        self.add_button.setVisible(not active)
        self.edit_group.setVisible(active)
        self.display_group.setVisible((not active) and has_bounds)

    def __update_samples_action(self):
        has_bounds = len(self.configuration.saved_bounds) > 0
        self.run_samples_button.setVisible(has_bounds)
        self.run_samples_button.setEnabled(has_bounds)

    def __update_display_bounds(self):
        index = self.configuration.selected_bounds_index
        if index < 0 or index >= len(self.configuration.saved_bounds):
            self.display_group.setTitle("Bounds")
            self.display_group.set_values(None)
            self.__update_samples_action()
            return
        bounds = self.configuration.saved_bounds[index]
        self.display_group.setTitle(f"Bounds {index + 1:02d}")
        self.display_group.set_values(bounds.get_values())
        self.__update_samples_action()
        self.__update_sample_results()

    def __update_sample_results(self):
        index = self.configuration.selected_bounds_index
        if index < 0 or index >= len(self.configuration.saved_bounds):
            self.sample_metrics.set_result(None)
            self.sample_metrics.setVisible(False)
            self.full_results_button.setEnabled(False)
            self.full_results_button.setVisible(False)
            return
        bounds = self.configuration.saved_bounds[index]
        result = bounds.get_sample()
        # Force a full widget refresh to avoid stale/corrupted summary rendering
        # when samples are rerun while dialogs are opened/closed.
        self.sample_metrics.setVisible(False)
        self.sample_metrics.set_result(None)
        if result is not None:
            self.sample_metrics.set_result(result)
            self.sample_metrics.updateGeometry()
            self.sample_metrics.update()
            self.sample_metrics.setVisible(True)
        self.full_results_button.setEnabled(result is not None)
        self.full_results_button.setVisible(result is not None)

    def __on_full_results_clicked(self):
        index = self.configuration.selected_bounds_index
        if index < 0 or index >= len(self.configuration.saved_bounds):
            return
        result = self.configuration.saved_bounds[index].get_sample()
        if result is None:
            return
        parent = self.controller.current_network_view
        parent.open_dialog(SampleResultsDialog(parent.close_dialog, result))
