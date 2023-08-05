from genetic.Individual import Individual
import random
import params
import copy
import statistics
import matplotlib.pyplot as plt
import numpy as np
import os
import math

def init_population():
    population = [Individual() for _ in range(params.RUN["population_size"])]
    return population

def sort_by_fitness(population):
    population.sort(key=lambda x: x.fitness)

def select_for_tournament(population):
    return random.sample(population, params.PRT_SEL["num_indiv_selected"])

def do_tournament(population):
    parents = []
    for _ in range(params.PRT_SEL["number_of_parents"]):
        selected = select_for_tournament(population)
        sort_by_fitness(selected)
        for elem in selected:
            if elem not in parents:
                parents.append(elem)
                break
    return parents

def discrete_choice(parents, gene_idx):
    prob = 1 / len(parents)
    return (random.choice(parents).gene[gene_idx], prob)

def aritimetic_combination(parent1, parent2, gene_idx):
    return params.CROSSOVER["alpha"] * parent1[gene_idx] + (1 - params.CROSSOVER["alpha"]) * parent2[gene_idx]

def base_crossover(parents, cut_point, quant_to_modify, offspring_idx):

    num_parents = len(parents)
    offspring = copy.deepcopy(parents[offspring_idx % num_parents].gene)

    if (num_parents < 2):
        print("At least 2 parents needed!")
        return
    elif (num_parents == 2):
        
        for q in range(quant_to_modify):
            if (offspring_idx % 2 == 0):
                offspring[cut_point + q] = aritimetic_combination(parents[1].gene,parents[0].gene,cut_point + q)
            else:
                offspring[cut_point + q] = aritimetic_combination(parents[0].gene,parents[1].gene,cut_point + q)
        offspring = Individual(offspring)
    else: #num_parents > 2
        
        for q in range(quant_to_modify):
            gene, _ = discrete_choice(parents,cut_point + q)
            offspring[cut_point + q] = gene
        offspring = Individual(offspring)
        
    return offspring

def simple_crossover(parents):
    offspring = []
    quant_to_modify = 1
    num_children = params.CROSSOVER["offspring_size"]

    for i in range(num_children):
        cut_point = random.randint(0,29)
        
        offspring.append(base_crossover(parents, cut_point, quant_to_modify, i))

    return offspring

def normal_crossover(parents):
    offspring = []
    num_children = params.CROSSOVER["offspring_size"]

    for i in range(num_children):
        cut_point = random.randint(0,29) + 1
        quant_to_modify = len(parents[0].gene) - cut_point

        offspring.append(base_crossover(parents, cut_point, quant_to_modify, i))
        
    return offspring

def complete_crossover(parents):
    offspring = []
    quant_to_modify = len(parents[0].gene)
    cut_point = 0
    num_children = params.CROSSOVER["offspring_size"]

    for i in range(num_children):
        offspring.append(base_crossover(parents, cut_point, quant_to_modify, i))

    return offspring

def crossover(parents):
    if (params.CROSSOVER["type"] == "simple"):
        return simple_crossover(parents)
    elif (params.CROSSOVER["type"] == "normal"):
        return normal_crossover(parents)
    elif (params.CROSSOVER["type"] == "complete"):
        return complete_crossover(parents)
    else:
        print("This crossover type does not exist or was not implemented.")

def mutate(offspring):
    new_offspring = []
    for individual in offspring:
        gene = copy.deepcopy(individual.gene)
        gene_len = len(gene)
        selected_gene_idx = random.randrange(gene_len)
        gene[selected_gene_idx] = random.uniform(params.FUNCTION["f_lo"], params.FUNCTION["f_hi"])
        # gene = [random.uniform(params.FUNCTION["f_lo"], params.FUNCTION["f_hi"]) for i in range(30)]
        individual.set_gene(gene)
        new_offspring.append(individual)
    return new_offspring

def survivor_selection(population):
    sort_by_fitness(population)
    return population[:-params.CROSSOVER["offspring_size"]]

def print_pop_comparison(old_pop, population):
    #### Pop comparision print
    print("=============================================")

    sort_by_fitness(old_pop)
    for pop in old_pop:
        print(pop,end=" | ")

    print("\n=============================================")
    
    for pop in population:
        print(pop,end=" | ")

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

def save_statistic(avg_fitness_iter, best_indiv_iter, std_fitness, run_num, title="Metrics per iteration"):
    plt.plot(avg_fitness_iter, label = 'Avg', linestyle='-')
    plt.plot(std_fitness, label= "Std",linestyle='-')
    plt.plot(best_indiv_iter, label= "Best",linestyle='-')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.title(title)
    plt.legend()

    path_abs = os.path.abspath(os.getcwd())
    path_title_list = title.split('_')
    path = os.path.join(path_abs, f"/data/{''.join(path_title_list)}")
    plt.savefig(path)

def pop_avg_fitness(population):
    fitness_pop = [ind.fitness for ind in population]
    return statistics.fmean(fitness_pop)

def pop_individual_fitness(population):
    return [pop.fitness for pop in population]

def execution():
    pass

def run_ga():
    # Initialization of population
    population = init_population()
    old_pop = copy.deepcopy(population)

    # Loop variables
    best_individual = population[0]
    iter = 0

    # Statistics to plot
    best_individuals = [best_individual.fitness]
    avg_fitness = [pop_avg_fitness(population)]
    std_fitness = [np.std(pop_individual_fitness(population))]

    # Statistics to plot before inferior to 1
    best_individuals_begin = [best_individual.fitness]
    avg_fitness_begin = [pop_avg_fitness(population)]
    std_fitness_begin = [np.std(pop_individual_fitness(population))]

    it_with_same_fitness = 0

    last_avg_fitness = pop_avg_fitness(population)
    forced_mutation = False

    while(iter < params.RUN["max_iterations"] and best_individual.fitness >= params.FUNCTION["global_min"]):
        if(iter % params.RUN["print_step"] == 0):
            print(f"({iter}th iter)  {[str(pop) for pop in population]}\n")

        # Selection of parents
        parents = do_tournament(population)

        # Crossover and generation of offspring
        offspring = []
        crossover_num = random.random()
        if (crossover_num <= params.CROSSOVER["chance"]):
            offspring += crossover(parents)
        else:
            offspring_size = params.CROSSOVER["offspring_size"]
            n_parents = params.PRT_SEL["number_of_parents"]
            
            for _ in range(math.ceil(offspring_size / n_parents)):
                offspring += [copy.deepcopy(parent) for parent in parents]
            
            if(len(offspring) != offspring_size):
                offspring = offspring[:offspring_size]

        # print(f"{iter} {crossover_num <= params.CROSSOVER['chance']} {len(offspring)} {len(population)}")

        # Mutation
        mutation_num = random.random()
        if(mutation_num <= params.MUTATION["prob"]):
            offspring = mutate(offspring)

        # Append offspring and select survivors
        population.extend(offspring)
        population = survivor_selection(population)
        
        # Increase iter counter
        iter += 1

        # Update Metrics
        best_individual = population[0]
        best_individuals.append(best_individual.fitness)
        avg_fitness.append(pop_avg_fitness(population))
        std_fitness.append(np.std(pop_individual_fitness(population)))

        # Check force mutation
        current_avg_fitness = avg_fitness[-1]
        if(abs(current_avg_fitness - last_avg_fitness) <= params.MUTATION["force_mutate_tol"] and not forced_mutation):
            it_with_same_fitness += 1
            if (it_with_same_fitness >= params.MUTATION["force_mutate_it"]):
                params.MUTATION["prob"] = params.MUTATION["forced_prob"]
                print("FORCING MUTATION")
                forced_mutation = True
        else:
            it_with_same_fitness = 0
        last_avg_fitness = current_avg_fitness

        # Update "begin" metrics
        if (current_avg_fitness >= 1):
            avg_fitness_begin.append(current_avg_fitness)
        # if (best_individual.fitness >= 1):
            best_individuals_begin.append(best_individual.fitness)
            std_fitness_begin.append(np.std(pop_individual_fitness(population)))

    # Show statistics
    print_pop_comparison(old_pop, population)
    plot_statistic(avg_fitness, best_individuals, std_fitness, 1)
    plot_statistic(avg_fitness_begin, best_individuals_begin, std_fitness_begin, 1, title="Begin metrics per iteration")
