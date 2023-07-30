import sys

sys.path.append("..")

import random
import params
from functions.FitnessFunction import FitnessFunction

class Individual_EE:
    def __init__(self, features=None, step=None):
        if features is None:
            # self.features = np.random(lo_range, hi_range, 30)
            self.features = [random.uniform(params.FUNCTION["f_lo"], params.FUNCTION["f_hi"]) for i in range(30)]
        else:
            self.features = features

        if step is None:
            self.step = random.random()*5
        else:
            self.step = step

        self.fitness_function = FitnessFunction()
        self.fitness = self.calc_fitness()

    def calc_fitness(self):
        fit = self.fitness_function.calculate(self.features)
        return fit

    def set_gene(self, new_features, step):
        self.features = new_features
        self.step = step
        self.fitness = self.calc_fitness()

    def get_gene(self):
        return f"|| fts: {self.features} | stp: {self.step} ||" 

    def __str__(self):
        return str(round(self.fitness,5))
