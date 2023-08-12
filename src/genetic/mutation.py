import copy, random
import params

def mutate(offspring):
    new_offspring = []
    for individual in offspring:
        gene = copy.deepcopy(individual.gene)
        gene_len = len(gene)
        selected_gene_idx = random.randrange(gene_len)
        gene[selected_gene_idx] = random.uniform(params.FUNCTION["f_lo"], params.FUNCTION["f_hi"])
        individual.set_gene(gene)
        new_offspring.append(individual)
    return new_offspring