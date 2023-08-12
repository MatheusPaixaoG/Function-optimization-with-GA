import math, numpy as np, sys
sys.path.append("..")


import params
from generic_utils import *
from evolutive.mutation import *
from evolutive.crossover import *
from evolutive.population_utils import *


# Adjust learning rate to params.EE["step_method"]
learning_rate = None
learning_rate_global = None

def print_n_save_statistics(pop, best_ind, iter, best_iter, avg_fit, best_inds, std_fit):
    if(iter % params.RUN["print_step"] == 0):
        print_txt = (f"({iter}th iter)  {[str(p) for p in pop]}\n")
        print(print_txt)
        print(f"BEST FITNESS UNTIL NOW: {best_ind.fitness} FOUND IN ITER {best_iter}")

        # Show statistics
        save_statistic(avg_fit, best_inds, std_fit, iter)

def generate_offspring(population):
    offspring = []
    for _ in range(params.CROSSOVER["offspring_size"]):
        offspring_features, parents = crossover(population, "local_middle", step_crossover=False)

        if(params.EE["step_method"] == "single"):
            offspring_step = crossover(population, "local_middle", parents, step_crossover=True)[0]
        else:
            offspring_step, _ = crossover(population, "local_middle", parents, step_crossover=True)

        offspring_ind = step_method_based_individual(offspring_features, offspring_step)
        
        offspring.append(offspring_ind)
    return offspring

def mutate_and_select(population, offspring):
    new_pop = population
    new_offspring = offspring
    if params.EE["survivor_selection_method"] == "elitist":
        new_pop += new_offspring
        new_pop = mutate(new_pop, learning_rate, learning_rate_global)
        new_pop = survivor_selection(new_pop)
    elif params.EE["survivor_selection_method"] == "only_offspring":
        new_offspring = mutate(new_offspring, learning_rate, learning_rate_global)
        new_pop = survivor_selection(new_offspring)
    else: 
        print("Survivor selection method not implemented")
    return new_pop, new_offspring

def set_learning_rates():
    global learning_rate
    global learning_rate_global

    if params.EE["step_method"] == "single":
        learning_rate = params.EE["learning_rate_modifier"] * (1 / math.sqrt(30))
    elif params.EE["step_method"] == "multi":
        learning_rate = params.EE["learning_rate_modifier"] * (1 / math.sqrt(2 * math.sqrt(30)))
        learning_rate_global = params.EE["learning_rate_global_modifier"] * (1 / math.sqrt(2*30))
    else:
        print("Step method not implemented.")

def execution(execution_num=1, function='ackley'):
    global mutation_epsilon
    population = init_population()
    sort_by_fitness(population)

    # Update Learning Rates
    set_learning_rates()

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
        print_n_save_statistics(population, best_global_individual, iter, best_iter, avg_fitness, best_individuals, std_fitness)

        offspring = generate_offspring(population)
        
        population, offspring = mutate_and_select(population, offspring)

        curr_avg_fitness = pop_avg_fitness(population)
        if (curr_avg_fitness < 2):
            mutation_epsilon = 1e-3
            set_learning_rates()
        elif (curr_avg_fitness < 1e-2):
            mutation_epsilon = 1e-5
            params.EE["learning_rate_modifier"] = 1e-2
            params.EE["learning_rate_global_modifier"] = 1e-2
            set_learning_rates()

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
