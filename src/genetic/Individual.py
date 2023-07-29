import sys

sys.path.append("..")

from functions.FitnessFunction import FitnessFunction
import random
import params


class Individual:
    def __init__(self, gene=None):
        if gene is None:
            # self.gene = np.random(lo_range, hi_range, 30)
            self.gene = [random.uniform(params.FUNCTION["f_lo"], params.FUNCTION["f_hi"]) for i in range(30)]
        else:
            self.gene = gene
        self.fitness_function = FitnessFunction()
        self.fitness = self.calc_fitness()

    def calc_fitness(self):
        fit = self.fitness_function.calculate(self.gene)
        return fit

    def set_gene(self, new_gene):
        self.gene = new_gene
        self.fitness = self.calc_fitness()

    def __str__(self):
        return str(round(self.fitness,5))
