from functions.Function import Function
import math


class Rastrigin(Function):
    def __init__(self):
        super().__init__()

    def rastrigin(self, point):
        return 10 * len(point) + sum(
            [(i**2) - 10 * math.cos(2 * math.pi * i) for i in point]
        )

    def calculate(self, point):
        return self.rastrigin(point)

    def plot(self, points_limit):
        return super().plot(self.rastrigin, points_limit)
