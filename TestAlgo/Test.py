# box_ibp_autolirpa.py
ALGORITHM_NAME = "Test"
IS_DETERMINISTIC = True

import numpy as np


def calculate_output_bounds(onnx_model, input_bounds: np.ndarray):
    return [0,1],[1,2]
