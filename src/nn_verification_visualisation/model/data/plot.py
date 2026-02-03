from matplotlib.figure import Figure

class Plot:
    '''
    Data object for a single plot.
    :param name: the name of the plot.
    :param data: the data of the plot, provided by MatPlotlib.
    '''
    name: str
    data: Figure

    def __init__(self, name: str, data: Figure):
        self.name = name
        self.data = data