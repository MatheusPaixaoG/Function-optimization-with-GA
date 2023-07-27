from functions.Function import Function
import math


class Schwefel(Function):
    def __init__(self, lower_limit, upper_limit):
        super().__init__(lower_limit, upper_limit)

    def schwefel(self, point):
        return 418.9829 * len(point) - sum(
            [i * math.sin(math.sqrt(abs(i))) for i in point]
        )

    def calculate(self, point):
        return self.schwefel(point)

    def plot(self, points_limit):
        return super().plot(self.schwefel, points_limit)
