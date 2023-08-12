from genetic.Individual import Individual
import params
import random
from generic_utils import sort_by_fitness

def init_population():
  population = [Individual() for _ in range(params.RUN["population_size"])]
  return population

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

def survivor_selection(population):
  sort_by_fitness(population)
  return population[:-params.CROSSOVER["offspring_size"]]