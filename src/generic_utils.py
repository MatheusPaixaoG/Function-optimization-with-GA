import matplotlib.pyplot as plt
import os
import statistics
import sys

sys.path.append("..")

from datetime import datetime

def sort_by_fitness(population):
    population.sort(key=lambda x: x.fitness)

def pop_individual_fitness(population):
    return [pop.fitness for pop in population]

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