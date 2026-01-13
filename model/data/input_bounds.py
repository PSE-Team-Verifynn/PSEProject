from typing import Dict


class InputBounds:
    bounds: Dict[int, (float, float)]

    def __init__(self, bounds: Dict[int, (float, float)]):
        self.bounds = bounds