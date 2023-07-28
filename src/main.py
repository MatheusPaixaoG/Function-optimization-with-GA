from functions.FitnessFunction import Functions
from genetic.Individual import Individual
import copy
import random

def init_population(population_size, lower_limit, upper_limit, function_ID):
    population = [Individual(lower_limit, upper_limit, function_ID) for indiv in range(population_size)]
    return population

def sort_by_fitness(population):
    population.sort(key=lambda x: x.fitness)

def select_for_tournament(population, num_ind):
    return random.sample(population, num_ind)

def do_tournament(num_parents, size_of_selection):
    parents = []
    for i in range(num_parents):
        selected = select_for_tournament(population, size_of_selection)
        sort_by_fitness(selected)
        for elem in selected:
            if elem not in parents:
                parents.append(elem)
                break
    return parents

def crossover(parents, cut_point, lower_limit, upper_limit, function_ID, alpha, cross_type):
    if (cross_type == "simple"):
        return simple_crossover(parents, cut_point, lower_limit, upper_limit, function_ID, alpha)
    elif (cross_type == "normal"):
        return normal_crossover(parents, cut_point, lower_limit, upper_limit, function_ID, alpha)
    elif (cross_type == "complete"):
        return complete_crossover(parents, lower_limit, upper_limit, function_ID, alpha)
    else:
        print("This crossover type does not exist or was not implemented.")

def aritimetic_combination(parent1, parent2, gene, alpha):
    return alpha * parent1[gene] + (1 - alpha) * parent2[gene]

def base_crossover(parents, cut_point, quant_to_modify, lower_limit, upper_limit, function_ID, alpha):
    gene1 = copy.deepcopy(parents[0].gene)
    gene2 = copy.deepcopy(parents[1].gene)

    for q in range(quant_to_modify):
        gene1[cut_point + q] = aritimetic_combination(parents[1].gene,parents[0].gene,cut_point + q, alpha)
        gene2[cut_point + q] = aritimetic_combination(parents[0].gene,parents[1].gene,cut_point + q, alpha)

    offspring = []
    offspring.append(Individual(lower_limit, upper_limit, function_ID, gene1))
    offspring.append(Individual(lower_limit, upper_limit, function_ID, gene2))

    return offspring

def simple_crossover(parents, cut_point, lower_limit, upper_limit, function_ID, alpha):
    quant_to_modify = 1
    return base_crossover(parents, cut_point, quant_to_modify, lower_limit, upper_limit, function_ID, alpha)

def normal_crossover(parents, cut_point, lower_limit, upper_limit, function_ID, alpha):
    quant_to_modify = len(parents[0].gene) - cut_point
    return base_crossover(parents, cut_point, quant_to_modify, lower_limit, upper_limit, function_ID, alpha)

def complete_crossover(parents, lower_limit, upper_limit, function_ID, alpha):
    quant_to_modify = len(parents[0].gene)
    cut_point = 0
    return base_crossover(parents, cut_point, quant_to_modify, lower_limit, upper_limit, function_ID, alpha)

def mutate(individual, lower_limit, upper_limit):
    gene = copy.deepcopy(individual.gene)
    gene_len = len(gene)
    selected_gene_idx = random.randrange(gene_len)
    gene[selected_gene_idx] = round(random.uniform(lower_limit, upper_limit), 3)
    individual.gene = gene

def survivor_selection(population, remove_amount):
    sort_by_fitness(population)
    population[:-remove_amount]

if __name__ == "__main__":
    points_limit = 50000

    ack_low, ack_up = -32.768, 32.768
    rast_low, rast_up = -5.12, 5.12
    schw_low, schw_up = -500, 500
    ros_low, ros_up = -5, 10

    # Initialization of population
    population_size = 10
    population = init_population(population_size, ack_low, ack_up, Functions.ACKLEY)

    # Selection of parents
    number_of_parents = 2
    num_indiv_selected = 5
    parents = do_tournament(number_of_parents, num_indiv_selected)

    # Crossover and generation of offspring
    offspring = crossover(parents, 2, ack_low, ack_up, Functions.ACKLEY, 0.5, "complete")
    for i in range(len(parents)):
        print(parents[i])
        print(offspring[i])
        print("--------------------------------------------------")

    # Mutation (mutate the offspring, not population[0])
    mutate(offspring[0], ack_low, ack_up)
