from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QPushButton, QHBoxLayout, QDoubleSpinBox,QDataWidgetMapper

from nn_verification_visualisation.controller.input_manager.network_view_controller import NetworkViewController
from nn_verification_visualisation.model.data.network_verification_config import NetworkVerificationConfig
from nn_verification_visualisation.view.base_view.tab import Tab
from nn_verification_visualisation.view.network_view.network_widget import NetworkWidget

class NetworkPage(Tab):
    configuration: NetworkVerificationConfig
    controller: NetworkViewController
    input_count: int

    def __init__(self, controller: NetworkViewController, configuration: NetworkVerificationConfig):
        self.configuration = configuration
        self.controller = controller
        self.input_count = configuration.layers_dimensions[0]
        super().__init__(configuration.network.name)

    def get_content(self) -> QWidget:
        return NetworkWidget(self.configuration)

    def get_side_bar(self) -> QWidget:
        base = QWidget()

        title = QLabel("Input Layer Bounds")
        title.setObjectName("title")

        import_button = QPushButton("Import Bounds")
        import_button.setObjectName("background-button")
        import_button.clicked.connect(lambda: self.controller.load_bounds(self.configuration))

        bound_input_layout = QVBoxLayout()

        model = self.configuration.bounds

        self.mappers = [] #keeps mappers in memory

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
            min_input.valueChanged.connect(mapper.submit)

            max_input = QDoubleSpinBox(minimum=-999999, maximum=999999)
            max_input.setFixedWidth(75)
            max_input.setSingleStep(.1)
            mapper.addMapping(max_input, 1)
            max_input.valueChanged.connect(mapper.submit)

            row_layout.addWidget(label)
            row_layout.addWidget(min_input)
            row_layout.addWidget(max_input)

            mapper.setCurrentIndex(i)
            bound_input_layout.addLayout(row_layout)

            self.mappers.append(mapper)

        layout = QVBoxLayout()
        layout.addWidget(title)
        layout.addLayout(bound_input_layout)
        layout.addStretch()
        layout.addWidget(import_button)
        base.setLayout(layout)

        return base

    def on_changed(self, bounds_num: int, is_max: bool, new_val: float):
        self.configuration.bounds.bounds[bounds_num] = (new_val,new_val)
        print("{} changed to {}".format(bounds_num, new_val))
