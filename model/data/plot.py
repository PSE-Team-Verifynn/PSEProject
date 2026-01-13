from matplotlib.figure import Figure

class Plot:
    name: str
    data: Figure

    def __init__(self, name: str, data: Figure):
        self.name = name
        self.data = data