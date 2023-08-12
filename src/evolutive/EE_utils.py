import copy
import math
import numpy as np
import os
import params
import random
import sys

from datetime import datetime

sys.path.append("..")
from evolutive.Individual_EE import Individual_EE, Individual_EE_Multi
from generic_utils import *

# Algorithm parameters
population_size = 15
step_method = "single"
survivor_selection_method = "only_offspring"
offspring_size = 105
learning_rate_modifier = 7
learning_rate_global_modifier = 7 # Only used if step_method = "multi"
mutation_epsilon = 1

# Adjust learning rate to step_method
learning_rate = None
learning_rate_global = None

def set_learning_rates(learning_rate_modifier, learning_rate_global_modifier):
    global learning_rate
    global learning_rate_global

    if step_method == "single":
        learning_rate = learning_rate_modifier * (1 / math.sqrt(30))
    elif step_method == "multi":
        learning_rate = learning_rate_modifier * (1 / math.sqrt(2 * math.sqrt(30)))
        learning_rate_global = learning_rate_global_modifier * (1 / math.sqrt(2*30))
    else:
        print("Step method not implemented.")

def execution(execution_num=1, function='ackley'):
    global mutation_epsilon
    population = init_population()
    sort_by_fitness(population)

    # Update Learning Rates
    global learning_rate_modifier
    global learning_rate_global_modifier
    set_learning_rates(learning_rate_modifier, learning_rate_global_modifier)

    # Loop variables
    best_individual = population[0]
    iter = 0

    # Statistics to plot
    best_individuals = [best_individual.fitness]
    avg_fitness = [pop_avg_fitness(population)]
    std_fitness = [np.std(pop_individual_fitness(population))]
    best_global_individual = best_individual
    best_iter = 0

    while(iter < params.RUN["max_iterations"] and best_individual.fitness >= params.FUNCTION["global_min"]):        
        if(iter % params.RUN["print_step"] == 0):
            print_txt = (f"({iter}th iter)  {[str(pop) for pop in population]}\n")
            print(print_txt)
            print(f"BEST FITNESS UNTIL NOW: {best_global_individual.fitness} FOUND IN ITER {best_iter}")

            # Show statistics
            save_statistic(avg_fitness, best_individuals, std_fitness, iter)

        offspring = []
        for _ in range(offspring_size):
            offspring_features, parents = crossover(population, "local_middle", step_crossover=False)

            if(step_method == "single"):
                offspring_step = crossover(population, "local_middle", parents, step_crossover=True)[0]
            else:
                offspring_step, _ = crossover(population, "local_middle", parents, step_crossover=True)

            offspring_ind = step_method_based_individual(offspring_features, offspring_step)
            
            offspring.append(offspring_ind)
        
        if survivor_selection_method == "elitist":
            population += offspring
        
            population = mutate(population)
            population = survivor_selection(population)

        elif survivor_selection_method == "only_offspring":
            offspring = mutate(offspring)
            population = survivor_selection(offspring)

        else:
            print("Survivor selection method not implemented")

        curr_avg_fitness = pop_avg_fitness(population)
        if (curr_avg_fitness < 2):
            mutation_epsilon = 1e-3
            set_learning_rates(learning_rate_modifier, learning_rate_global_modifier)
        elif (curr_avg_fitness < 1e-2):
            mutation_epsilon = 1e-5
            learning_rate_modifier = 1e-2
            learning_rate_global_modifier = 1e-2
            set_learning_rates(learning_rate_modifier, learning_rate_global_modifier)

        sort_by_fitness(population)
        best_individual = population[0]

        iter += 1

        # Update Metrics
        best_individual = population[0]
        best_individuals.append(best_individual.fitness)
        avg_fitness.append(curr_avg_fitness)
        std_fitness.append(np.std(pop_individual_fitness(population)))

        if (best_individual.fitness < best_global_individual.fitness):
            best_global_individual = best_individual
            best_iter = iter-1

    # Show statistics
    save_statistic(avg_fitness, best_individuals, std_fitness, execution_num, function)

    # Return execution metrics
    converged = best_global_individual.fitness <= params.FUNCTION["global_min"]
    return curr_avg_fitness, std_fitness[-1], iter, converged, best_global_individual, best_iter

def run_ee(num_executions=1, function='ackley'):
    # Per execution metrics
    avg_fit_ex = []
    std_fit_ex = []
    n_iters_ex = []
    converged_ex = []
    best_ind_ex = []
    best_iter_ex = []

    for i in range(num_executions):
        print(f'\nEXECUTION {i+1} \n')
        
        # Run the algorithm and obtains the execution metrics
        avg_fit, std_fit, n_iters, converged, best_individual, best_iter = execution(i, function)
        
        # Stores the execution metrics
        avg_fit_ex.append(avg_fit)
        std_fit_ex.append(std_fit)
        n_iters_ex.append(n_iters)
        converged_ex.append(converged)
        best_ind_ex.append(best_individual.fitness)
        best_iter_ex.append(best_iter)

    # Show statistics
    avg_avg_fit = sum(avg_fit_ex)/num_executions
    avg_std_fit = sum(std_fit_ex)/num_executions
    avg_iters_ex = sum(n_iters_ex)/num_executions
    converged_perc = converged_ex.count(True)/num_executions
    avg_best_ind_ex = sum(best_ind_ex)/num_executions
    avg_best_iter_ex = sum(best_iter_ex)/num_executions

    save_avg_execution_metrics(avg_avg_fit, avg_std_fit, avg_iters_ex, converged_perc, avg_best_ind_ex, avg_best_iter_ex, function)

def step_method_based_individual(features, step):
    individual = None
    if(step_method == "single"):
        individual = Individual_EE(features, step)
    elif(step_method == "multi"):
        individual = Individual_EE_Multi(features, step)
    else:
        print("Step method not found")

    return individual

def init_population():
    population = []
    if(step_method == "single"):
        population = [Individual_EE() for _ in range(population_size)]
    elif(step_method == "multi"):
        population  = [Individual_EE_Multi() for _ in range(population_size)]
    return population

def parent_selection(population, n_parents = 2, allow_repetitions=False):
    if (allow_repetitions or n_parents > len(population)):
        return random.choices(population, n_parents)
    else:
        return random.sample(population, n_parents)

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

def mutate(population):
    new_population = []
    for individual in population:
        # Copying original values
        features = copy.deepcopy(individual.features)
        step = copy.deepcopy(individual.step)

        new_step = None

        # Mutating evolution step
        if (step_method == "single"):
            new_step = step * math.exp(learning_rate * random.gauss(0,1))
            if new_step < mutation_epsilon:
                new_step = mutation_epsilon
        elif (step_method == "multi"):
            new_step = []
            step_global = learning_rate_global * random.gauss(0,1)
            for i in range(len(step)):
                step_local = learning_rate * random.gauss(0,1)
                
                curr_step = step[i] * math.exp(step_global + step_local)
                if curr_step < mutation_epsilon:
                    curr_step = mutation_epsilon
                new_step.append(curr_step)
        else:
            print("This step type was not implemented.")


        # Mutating features
        if(step_method == "single"):
                new_features = [x + new_step * random.gauss(0,1) for x in features]
        elif (step_method == "multi"):
            new_features = [features[i] + (new_step[i] * random.gauss(0,1)) for i in range(len(features))]

        mutant_indv = step_method_based_individual(new_features, new_step)

        # Discard bad mutations
        if(mutant_indv.fitness > individual.fitness):
            new_population.append(mutant_indv)
        else:
            new_population.append(individual)
    
    return new_population

def survivor_selection(population):
    sort_by_fitness(population)
    return population[:population_size]
