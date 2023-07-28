import sys

sys.path.append("..")

from functions.FitnessFunction import Functions, FitnessFunction
import numpy as np


class Individual:
    def __init__(self, lo_range, hi_range, function_ID, gene=None):
        if gene is None:
            self.gene = np.round(np.random.uniform(lo_range, hi_range, 30), 3)
        else:
            self.gene = gene
        self.fitness_function = FitnessFunction(lo_range, hi_range)
        self.fitness = round(self.calc_fitness(function_ID), 4)

    def calc_fitness(self, function_ID):
        fit = self.fitness_function.calculate(self.gene, function_ID)
        return round(fit, 4)

    def __str__(self):
        return f"[Individual] {self.gene} {self.fitness}"
