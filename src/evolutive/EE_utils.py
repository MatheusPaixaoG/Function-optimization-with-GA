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
from evolutive.Individual_EE import Individual_EE, Individual_EE_Multi

# Algorithm parameters
population_size = 10
step_method = "multi"
offspring_size = 70  # 
learning_rate_modifier = 1
learning_rate_global_modifier = 1 # Only used if step_method = "multi"
mutation_epsilon = 0.1

# Adjust learning rate to step_method
learning_rate = None
learning_rate_global = None
if step_method == "single":
    learning_rate = learning_rate_modifier * (1 / math.sqrt(30))
elif step_method == "multi":
    learning_rate = learning_rate_modifier * (1 / math.sqrt(2 * math.sqrt(30)))
    learning_rate_global = learning_rate_global_modifier * (1 / math.sqrt(2*30))
else:
    print("Step method not implemented.")

def run_ee():
    global mutation_epsilon
    population = init_population()
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

            if(step_method == "single"):
                offspring_step = crossover(population, "local_middle", step_crossover=True)[0]
            else:
                offspring_step = crossover(population, "local_middle", step_crossover=True)

            offspring_ind = step_method_based_individual(offspring_features, offspring_step)
            
            offspring.append(offspring_ind)
        
        population += offspring
        
        population = mutate(population)
        population = survivor_selection(population)

        curr_avg_fitness = pop_avg_fitness(population)
        if (curr_avg_fitness < 2):
            mutation_epsilon = 0.001

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

def sort_by_fitness(population):
    population.sort(key=lambda x: x.fitness)

def pop_individual_fitness(population):
    return [pop.fitness for pop in population]

def parent_selection(population, n_parents = 2, allow_repetitions=False):
    if (allow_repetitions or n_parents > len(population)):
        return random.choices(population, n_parents)
    else:
        return random.sample(population, n_parents)

def crossover(population, parent_selection_mode, step_crossover=False):
    if(parent_selection_mode == "local_middle"):
        parents = parent_selection(population, n_parents=2, allow_repetitions=False)
        return local_middle_crossover(parents, step_crossover)
    
    elif(parent_selection_mode == "global_middle"):
        step_len = len(population[0].step)
        n_parents = step_len if step_crossover else 60
        
        parents = parent_selection(population, n_parents=n_parents, allow_repetitions=True)
        return global_middle_crossover(parents, step_crossover)

    elif(parent_selection_mode == "local_discrete"):
        parents = parent_selection(population, n_parents=2, allow_repetitions=False)
        return local_discrete_crossover(parents, step_crossover)
    
    elif(parent_selection_mode == "global_discrete"):
        step_len = len(population[0].step)
        n_parents = step_len if step_crossover else 60
        
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