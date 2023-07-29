from functions.Ackley import Ackley
from functions.Rastrigin import Rastrigin
from functions.Schwefel import Schwefel
from functions.Rosenbrock import Rosenbrock

import sys
sys.path.append("..")

import params

class FitnessFunction:
    def __init__(self):
        if params.FUNCTION["current_function"] == params.Functions.ACKLEY:
            self.function = Ackley()
        elif params.FUNCTION["current_function"] == params.Functions.RASTRIGIN:
            self.function = Rastrigin()
        elif params.FUNCTION["current_function"] == params.Functions.ROSENBROCK:
            self.function = Rosenbrock()
        elif params.FUNCTION["current_function"] == params.Functions.SCHWEFEL:
            self.function = Schwefel()
        else: 
            self.function = Ackley()

    def calculate(self, point):
        return self.function.calculate(point)

    def plot(self, points_limit):
        return self.function.plot(points_limit)
