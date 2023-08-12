import copy
import numpy as np
import random
from evolutive.population_utils import parent_selection

def crossover(population, parent_selection_mode, parents = None, step_crossover=False):
    if(parent_selection_mode == "local_middle"):
        if(parents == None):
            parents = parent_selection(population, n_parents=2, allow_repetitions=False)
        return (local_middle_crossover(parents, step_crossover), parents)
    
    elif(parent_selection_mode == "global_middle"):
        step_len = len(population[0].step)
        n_parents = step_len if step_crossover else 60

        if(parents == None):
            parents = parent_selection(population, n_parents=n_parents, allow_repetitions=True)
        return (global_middle_crossover(parents, step_crossover), parents)

    elif(parent_selection_mode == "local_discrete"):
        if(parents == None):
            parents = parent_selection(population, n_parents=2, allow_repetitions=False)
        return (local_discrete_crossover(parents, step_crossover), parents)
    
    elif(parent_selection_mode == "global_discrete"):
        step_len = len(population[0].step)
        n_parents = step_len if step_crossover else 60

        if(parents == None):
            parents = parent_selection(population, n_parents=n_parents, allow_repetitions=True)
        return (global_discrete_crossover(parents, step_crossover), parents)

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