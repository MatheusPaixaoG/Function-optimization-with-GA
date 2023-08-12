import params
import copy
from genetic.Individual import Individual
import random

class Crossover:
  def discrete_choice(self, parents, gene_idx):
    prob = 1 / len(parents)
    return (random.choice(parents).gene[gene_idx], prob)

  def aritimetic_combination(self, parent1, parent2, gene_idx):
    return params.CROSSOVER["alpha"] * parent1[gene_idx] + (1 - params.CROSSOVER["alpha"]) * parent2[gene_idx]
  
  def base_crossover(self, parents, cut_point, quant_to_modify, offspring_idx):
    num_parents = len(parents)
    offspring = copy.deepcopy(parents[offspring_idx % num_parents].gene)

    if (num_parents < 2):
        print("At least 2 parents needed!")
        return
    elif (num_parents == 2):
        
        for q in range(quant_to_modify):
            if (offspring_idx % 2 == 0):
                offspring[cut_point + q] = self.aritimetic_combination(parents[1].gene,parents[0].gene,cut_point + q)
            else:
                offspring[cut_point + q] = self.aritimetic_combination(parents[0].gene,parents[1].gene,cut_point + q)
        offspring = Individual(offspring)
    else: #num_parents > 2
        
        for q in range(quant_to_modify):
            gene, _ = self.discrete_choice(parents,cut_point + q)
            offspring[cut_point + q] = gene
        offspring = Individual(offspring)
        
    return offspring

  def simple_crossover(self, parents):
    offspring = []
    quant_to_modify = 1
    num_children = params.CROSSOVER["offspring_size"]

    for i in range(num_children):
        cut_point = random.randint(0,29)
        
        offspring.append(self.base_crossover(parents, cut_point, quant_to_modify, i))

    return offspring

  def normal_crossover(self, parents):
    offspring = []
    num_children = params.CROSSOVER["offspring_size"]

    for i in range(num_children):
        cut_point = random.randint(0,29) + 1
        quant_to_modify = len(parents[0].gene) - cut_point

        offspring.append(self.base_crossover(parents, cut_point, quant_to_modify, i))
        
    return offspring

  def complete_crossover(self, parents):
    offspring = []
    quant_to_modify = len(parents[0].gene)
    cut_point = 0
    num_children = params.CROSSOVER["offspring_size"]

    for i in range(num_children):
        offspring.append(self.base_crossover(parents, cut_point, quant_to_modify, i))

    return offspring

  def crossover(self, parents):
    if (params.CROSSOVER["type"] == "simple"):
        return self.simple_crossover(parents)
    elif (params.CROSSOVER["type"] == "normal"):
        return self.normal_crossover(parents)
    elif (params.CROSSOVER["type"] == "complete"):
        return self.complete_crossover(parents)
    else:
        print("This crossover type does not exist or was not implemented.")