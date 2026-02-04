import numpy as np

class Plot:
    '''
    Data object for a single plot.
    :param name: the name of the plot.
    :param data: output bounds data, shape (N, 2) as [lower, upper].
    '''
    name: str
    data: np.ndarray

    def __init__(self, name: str, data: np.ndarray):
        self.name = name
        self.data = data
