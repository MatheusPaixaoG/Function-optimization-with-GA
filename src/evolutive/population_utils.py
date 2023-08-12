import random

import params
from evolutive.Individual import *
from generic_utils import sort_by_fitness

def init_population():
    population = []
    if(params.EE["step_method"] == "single"):
        population = [Individual() for _ in range(params.RUN["population_size"])]
    elif(params.EE["step_method"] == "multi"):
        population  = [IndividualMulti() for _ in range(params.RUN["population_size"])]
    return population

def parent_selection(population, n_parents = 2, allow_repetitions=False):
    if (allow_repetitions or n_parents > len(population)):
        return random.choices(population, n_parents)
    else:
        return random.sample(population, n_parents)
    
def survivor_selection(population):
    sort_by_fitness(population)
    return population[:params.RUN["population_size"]]