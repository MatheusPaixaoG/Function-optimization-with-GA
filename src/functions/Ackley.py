from functions.Function import Function
import numpy as np


class Ackley(Function):
    def __init__(self, lower_limit, upper_limit):
        super().__init__(lower_limit, upper_limit)

    def ackley(self, point, a=20, b=0.2, c=2 * np.pi):
        d = len(point)
        # First, we calculate the summation terms
        fst_sum = sum(np.multiply(point, point))
        snd_sum = sum(np.cos(np.multiply(c, point)))
        # Then, we calculate each term that uses a summation
        srqt_term = -a * np.exp(-b * np.sqrt(fst_sum / d))
        cos_term = -np.exp(snd_sum / d)
        # Now, we can calculate the result of the function
        result = srqt_term + cos_term + a + np.exp(1)
        return result

    def calculate(self, point):
        return self.ackley(point)

    def plot(self, points_limit):
        return super().plot(self.ackley, points_limit)
