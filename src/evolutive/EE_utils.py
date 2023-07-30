import random
import params
import copy
import sys
import math

sys.path.append("..")
from evolutive.Individual_EE import Individual_EE

population_size = 10
learning_rate_modifier = 3
learning_rate = learning_rate_modifier * 1/(30**0.5)
mutation_epsilon = 0.01

def run_ee():
    population = [Individual_EE() for _ in range(10)]
    print([str(pop) for pop in population])

    parents = parent_selection(population)

    print()
    print([p.fitness for p in parents])

def init_population():
    population = [Individual_EE() for _ in range(population_size)]
    return population

def sort_by_fitness(population):
    population.sort(key=lambda x: x.fitness)

def parent_selection(population, n_parents = 2):
    parents = []
    for _ in range(n_parents):
        chosen = None

        while chosen == None or chosen not in population:
            chosen = random.choice(population)

        parents.append(chosen)
        
    return parents
        

def mutate(offspring):
    new_offspring = []
    for individual in offspring:
        # Copying original values
        features = copy.deepcopy(individual.features)
        step = copy.deepcopy(individual.step)

        # Mutating evolution step
        new_step = step * math.exp(learning_rate * random.gauss(0,1))
        if new_step < mutation_epsilon:
            new_step = mutation_epsilon

        # Mutating features
        new_features = [x + (new_step * random.gauss(0,1)) for x in features]

        mutant_indv = Individual_EE(new_features,new_step)

        # Discard bad mutations
        if(mutant_indv.fitness > individual.fitness):
            new_offspring.append(mutant_indv)
            print("Good mutation")
        else:
            new_offspring.append(individual)
            print("Bad mutation")
    
    return new_offspring