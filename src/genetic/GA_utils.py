from genetic.Individual import Individual
import random
import params
import copy

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

def aritimetic_combination(parent1, parent2, gene_idx):
    return params.CROSSOVER["alpha"] * parent1[gene_idx] + (1 - params.CROSSOVER["alpha"]) * parent2[gene_idx]

def base_crossover(parents, cut_point, quant_to_modify):
    gene1 = copy.deepcopy(parents[0].gene)
    gene2 = copy.deepcopy(parents[1].gene)

    for q in range(quant_to_modify):
        gene1[cut_point + q] = aritimetic_combination(parents[1].gene,parents[0].gene,cut_point + q)
        gene2[cut_point + q] = aritimetic_combination(parents[0].gene,parents[1].gene,cut_point + q)

    offspring = []
    offspring.append(Individual(gene1))
    offspring.append(Individual(gene2))
    return offspring

def simple_crossover(parents, cut_point):
    quant_to_modify = 1
    return base_crossover(parents, cut_point, quant_to_modify)

def normal_crossover(parents, cut_point):
    quant_to_modify = len(parents[0].gene) - cut_point
    return base_crossover(parents, cut_point, quant_to_modify)

def complete_crossover(parents):
    quant_to_modify = len(parents[0].gene)
    cut_point = 0
    return base_crossover(parents, cut_point, quant_to_modify)

def crossover(parents, cut_point):
    if (params.CROSSOVER["type"] == "simple"):
        return simple_crossover(parents, cut_point)
    elif (params.CROSSOVER["type"] == "normal"):
        return normal_crossover(parents, cut_point)
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
    return population[:-params.SVV_SEL["offspring_size"]]

def print_pop_comparison(old_pop, population):
    #### Pop comparision print
    print("=============================================")

    sort_by_fitness(old_pop)
    for pop in old_pop:
        print(pop,end=" | ")

    print("\n=============================================")
    
    for pop in population:
        print(pop,end=" | ")

def run_ga():
    # Initialization of population
    population = init_population()

    old_pop = copy.deepcopy(population)

    # Loop variables
    best_individual = population[0]
    iter = 0
    
    while(iter < params.RUN["max_iterations"] and best_individual.fitness >= params.FUNCTION["global_min"]):

        if(iter % params.RUN["print_step"] == 0):
            print(f"({iter}th iter)  {[str(pop) for pop in population]}\n")

        # Selection of parents
        parents = do_tournament(population)

        # Crossover and generation of offspring
        offspring = []
        crossover_num = random.random()
        if (crossover_num <= params.CROSSOVER["chance"]):
            cut_point = random.randint(0,29)
            offspring = crossover(parents, cut_point)
        else:
            offspring += [copy.deepcopy(parent) for parent in parents]

        # Mutation
        mutation_num = random.random()
        if(mutation_num <= params.MUTATION["prob"]):
            offspring = mutate(offspring)

        # Append offspring and select survivors
        population += offspring
        population = survivor_selection(population)
        
        best_individual = population[0]
        iter += 1

    print_pop_comparison(old_pop, population)