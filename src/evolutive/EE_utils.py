import random
import params
import copy
import sys
import math
import numpy as np
import statistics
import matplotlib.pyplot as plt
import numpy as np
import os
import math
from datetime import datetime

sys.path.append("..")
from evolutive.Individual_EE import Individual_EE

population_size = 10
learning_rate_modifier = 10
learning_rate = learning_rate_modifier * 1/(30**0.5)
mutation_epsilon = 0.1
offspring_size = 14


def run_ee():
    global mutation_epsilon
    population = [Individual_EE() for _ in range(population_size)]
    sort_by_fitness(population)

    # Loop variables
    best_individual = population[0]
    iter = 0

    # Statistics to plot
    best_individuals = [best_individual.fitness]
    avg_fitness = [pop_avg_fitness(population)]
    std_fitness = [np.std(pop_individual_fitness(population))]

    while(iter < params.RUN["max_iterations"] and best_individual.fitness >= params.FUNCTION["global_min"]):
        if(iter % params.RUN["print_step"] == 0):
            print(f"({iter}th iter)  {[str(pop) for pop in population]}\n")

        offspring = []
        for _ in range(offspring_size):
            offspring_features = crossover(population, "local_middle", step_crossover=False)
            offspring_step = crossover(population, "local_middle", step_crossover=True)[0]
            offspring_ind = Individual_EE(offspring_features, offspring_step)
            offspring.append(offspring_ind)
        
        population += offspring
        
        population = survivor_selection(mutate(population))
        curr_avg_fitness = pop_avg_fitness(population)
        if (curr_avg_fitness < 2):
            mutation_epsilon = 0.01

        sort_by_fitness(population)
        best_individual = population[0]

        iter += 1

        # Update Metrics
        best_individual = population[0]
        best_individuals.append(best_individual.fitness)
        avg_fitness.append(curr_avg_fitness)
        std_fitness.append(np.std(pop_individual_fitness(population)))

    # Show statistics
    save_statistic(avg_fitness, best_individuals, std_fitness, 1)

def init_population():
    population = [Individual_EE() for _ in range(population_size)]
    return population

def sort_by_fitness(population):
    population.sort(key=lambda x: x.fitness)

def pop_individual_fitness(population):
    return [pop.fitness for pop in population]

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
        else:
            new_population.append(individual)
    
    return new_population

def survivor_selection(population):
    sort_by_fitness(population)
    return population[:-offspring_size]

def pop_avg_fitness(population):
    fitness_pop = [ind.fitness for ind in population]
    return statistics.fmean(fitness_pop)

def plot_statistic(avg_fitness_iter, best_indiv_iter, std_fitness, title="Metrics per iteration"):
    plt.plot(avg_fitness_iter, label = 'Avg', linestyle='-')
    plt.plot(std_fitness, label= "Std",linestyle='-')
    plt.plot(best_indiv_iter, label= "Best",linestyle='-')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.title(title)
    plt.legend()
    plt.show()

def save_statistic(avg_fitness_iter, best_indiv_iter, std_fitness, execution_num=1, title="Metrics per iteration"):
    plt.figure()
    plt.plot(avg_fitness_iter, label = 'Avg', linestyle='-')
    plt.plot(std_fitness, label= "Std",linestyle='-')
    plt.plot(best_indiv_iter, label= "Best",linestyle='-')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.title(title)
    plt.legend()

    curr_datetime = datetime.now().strftime('%m_%d_%H_%M_%S')

    path = os.path.join(os.getcwd(),"data",f"{curr_datetime}_{title + ' ' + str(execution_num)}")
    print(path)
    plt.savefig(path)

def save_avg_execution_metrics(avg_fit, std_fit, n_iters, perc_converged):
    curr_datetime = datetime.now().strftime('%m_%d_%H_%M_%S')
    path = os.path.join(os.getcwd(),"data",f"execution_metrics_{curr_datetime}.txt")

    with open(path, "w") as file:
        file_txt = f"Avg Fitness {avg_fit} \nStd Fitness {std_fit} \nNum. of iterations {n_iters} \nPerc. converged {perc_converged}" 
        file.write(file_txt)