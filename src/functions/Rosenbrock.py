from functions.Function import Function


class Rosenbrock(Function):
    def __init__(self, lower_limit, upper_limit):
        super().__init__(lower_limit, upper_limit)

    def rosenbrock(self, point):
        sum_list = []
        for i in range(len(point) - 1):
            sum_list.append(
                100 * (point[i + 1] - point[i] ** 2) ** 2 + (point[i] - 1) ** 2
            )
        return sum(sum_list)

    def calculate(self, point):
        return self.rosenbrock(point)

    def plot(self, points_limit):
        return super().plot(self.rosenbrock, points_limit)
