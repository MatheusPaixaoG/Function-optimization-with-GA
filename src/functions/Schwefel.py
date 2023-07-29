from functions.Function import Function
import math


class Schwefel(Function):
    def __init__(self):
        super().__init__()

    def schwefel(self, point):
        return 418.9829 * len(point) - sum(
            [i * math.sin(math.sqrt(abs(i))) for i in point]
        )

    def calculate(self, point):
        return self.schwefel(point)

    def plot(self, points_limit):
        return super().plot(self.schwefel, points_limit)
