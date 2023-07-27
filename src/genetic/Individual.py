import numpy as np

class Individual:
    def __init__(self, lo_range, hi_range):
        self.gene = (np.random(lo_range,hi_range) for i in range(30))