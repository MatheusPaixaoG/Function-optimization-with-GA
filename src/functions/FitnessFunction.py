from functions.Ackley import Ackley
from functions.Rastrigin import Rastrigin
from functions.Schwefel import Schwefel
from functions.Rosenbrock import Rosenbrock

from enum import Enum


class Functions(Enum):
    ACKLEY = 0
    RASTRIGIN = 1
    SCHWEFEL = 2
    ROSENBROCK = 3


class FitnessFunction:
    def __init__(self, lower_limit, upper_limit):
        self.ackley = Ackley(lower_limit, upper_limit)
        self.rastringin = Rastrigin(lower_limit, upper_limit)
        self.rosenbrock = Rosenbrock(lower_limit, upper_limit)
        self.schwefel = Schwefel(lower_limit, upper_limit)

    def calculate(self, point, function_ID):
        if function_ID == Functions.ACKLEY:
            return self.ackley.calculate(point)
        elif function_ID == Functions.RASTRIGIN:
            return self.rastringin.calculate(point)
        elif function_ID == Functions.SCHWEFEL:
            return self.schwefel.calculate(point)
        elif function_ID == Functions.ROSENBROCK:
            return self.rosenbrock.calculate(point)
        else:
            print("MANDOU UM NUMERO ERRADO")

    def plot(self, function_ID, points_limit):
        if function_ID == Functions.ACKLEY:
            return self.ackley.plot(points_limit)
        elif function_ID == Functions.RASTRIGIN:
            return self.rastringin.plot(points_limit)
        elif function_ID == Functions.SCHWEFEL:
            return self.schwefel.plot(points_limit)
        elif function_ID == Functions.ROSENBROCK:
            return self.rosenbrock.plot(points_limit)
        else:
            print("MANDOU UM NUMERO ERRADO")
