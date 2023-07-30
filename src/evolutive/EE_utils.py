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
    population = [Individual_EE() for _ in range(2)]
    
    for pop in population:
        print(f"f: {[f for f in pop.features]} | step: {pop.step} | fit: {pop.fitness}")
        print()

    ofspring = crossover(population, "biologic")
    print()

    print(f"f: {[f for f in ofspring.features]} | step: {ofspring.step} | fit: {ofspring.fitness}")


def init_population():
    population = [Individual_EE() for _ in range(population_size)]
    return population

def sort_by_fitness(population):
    population.sort(key=lambda x: x.fitness)

def parent_selection(population, n_parents = 2, allow_repetitions=False):
    parents = []
    for _ in range(n_parents):
        chosen = random.choice(population)

        if not allow_repetitions:
            while chosen in parents:
                chosen = random.choice(population)

        parents.append(chosen)
        
    return parents

def crossover(population, parent_selection_mode):
    if(parent_selection_mode == "biologic"):
        parents = parent_selection(population, n_parents=2, allow_repetitions=False)
        return biologic_crossover(parents)
    
    elif(parent_selection_mode == "per-gene"):
        parents = parent_selection(population, n_parents=60, allow_repetitions=True)
        return per_gene_crossover(population)

def biologic_crossover(parents):
    feat1 = copy.deepcopy(parents[0].features)
    feat2 = copy.deepcopy(parents[1].features)
    offspring_feats = []

    for i in range(0,len(feat2)):
        gene = (feat1[i] + feat2[i])/2
        offspring_feats.append(gene)

    offspring_step = (parents[0].step + parents[1].step)/2

    return Individual_EE(offspring_feats, offspring_step)

def per_gene_crossover(parents):
    pass

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