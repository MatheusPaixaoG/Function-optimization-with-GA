import random
import params
import copy
import sys
import math
import numpy as np

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

    ofspring_features = crossover(population, "global_discrete", step_crossover=False)
    ofspring_step = crossover(population, "global_discrete", step_crossover=True)[0]
    ofspring = Individual_EE(ofspring_features, ofspring_step)

    print()
    print(f"f: {[f for f in ofspring.features]} | step: {ofspring.step} | fit: {ofspring.fitness}")

    mutate(population)

    print()
    #print(f"f: {[f for f in population.features]} | step: {population.step} | fit: {population.fitness}")
    for pop in population:
        print(pop)


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

def crossover(population, parent_selection_mode, step_crossover=False):
    if(parent_selection_mode == "local_middle"):
        parents = parent_selection(population, n_parents=2, allow_repetitions=False)
        return local_middle_crossover(parents, step_crossover)
    
    elif(parent_selection_mode == "global_middle"):
        n_parents = 2 if step_crossover else 60
        parents = parent_selection(population, n_parents=n_parents, allow_repetitions=True)
        return global_middle_crossover(parents, step_crossover)

    elif(parent_selection_mode == "local_discrete"):
        parents = parent_selection(population, n_parents=2, allow_repetitions=False)
        return local_discrete_crossover(parents, step_crossover)
    
    elif(parent_selection_mode == "global_discrete"):
        n_parents = 2 if step_crossover else 60
        parents = parent_selection(population, n_parents=n_parents, allow_repetitions=True)
        return global_discrete_crossover(parents, step_crossover)

def local_middle_crossover(parents, step_crossover):
    parent_var1 = parents[0].get_gene_or_step(step_crossover)
    parent_var2 = parents[1].get_gene_or_step(step_crossover)

    offspring_vars = np.divide(np.array(parent_var1) + np.array(parent_var2), 2)
    offspring_vars = offspring_vars.tolist()

    return offspring_vars

def global_middle_crossover(parents, step_crossover):
    offspring_vars = []
    middle_index = int(len(parents)/2)

    for i in range(0,middle_index):
        p1 = parents[i]
        p2 = parents[i+middle_index]

        parent_var1 = p1.get_gene_or_step(step_crossover)
        parent_var2 = p2.get_gene_or_step(step_crossover)

        gene_i = (parent_var1[i] + parent_var2[i])/2
        offspring_vars.append(gene_i)

    return offspring_vars

def local_discrete_crossover(parents, step_crossover):
    parent_var1 = parents[0].get_gene_or_step(step_crossover)
    parent_var2 = parents[1].get_gene_or_step(step_crossover)

    parent_var1 = copy.deepcopy(parent_var1)
    parent_var2 = copy.deepcopy(parent_var2)
    parent_vars = [parent_var1, parent_var2]
    offspring_vars = []

    for i in range(0,len(parent_var2)):
        offspring_vars.append(random.choice(parent_vars)[i])

    return offspring_vars

def global_discrete_crossover(parents, step_crossover):
    offspring_vars = []
    middle_index = int(len(parents)/2)

    for i in range(0,middle_index):
        p1 = parents[i]
        p2 = parents[i+middle_index]

        parent_var1 = p1.get_gene_or_step(step_crossover)
        parent_var2 = p2.get_gene_or_step(step_crossover)

        parent_var1 = copy.deepcopy(parent_var1)
        parent_var2 = copy.deepcopy(parent_var2)
        parent_vars = [parent_var1, parent_var2]

        gene_i = random.choice(parent_vars)[i]
        offspring_vars.append(gene_i)

    return offspring_vars

def mutate(population):
    new_population = []
    for individual in population:
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
            new_population.append(mutant_indv)
            print("Good mutation")
        else:
            new_population.append(individual)
            print("Bad mutation")
    
    return new_population