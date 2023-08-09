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

def save_statistic(avg_fitness_iter, best_indiv_iter, std_fitness, execution_num=1, title="Metrics per iteration", function='ackley'):
    save_best_avg_statistics(best_indiv_iter, avg_fitness_iter, title, execution_num, function)
    save_avg_std_statistics(std_fitness, title, execution_num, function)
    plt.close("all") # Close all figures to save RAM

def save_best_avg_statistics(best_indiv_iter, avg_fitness_iter, title, execution_num, function):
    plt.figure()
    plt.plot(best_indiv_iter, label= "Best",linestyle='-')
    plt.plot(avg_fitness_iter, label = 'Avg', linestyle='-')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.title(title)
    plt.legend()

    curr_datetime = datetime.now().strftime('%m_%d_%H_%M_%S')
    path = os.path.join(os.getcwd(),"data",f"{curr_datetime}_{title + '_best_avg_' + str(execution_num) } ({function})")
    plt.savefig(path)

def save_avg_std_statistics(std_fitness, title, execution_num, function):
    plt.figure()
    plt.plot(std_fitness, label= "Std",linestyle='-')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.title(title)
    plt.legend()

    curr_datetime = datetime.now().strftime('%m_%d_%H_%M_%S')
    path = os.path.join(os.getcwd(),"data",f"{curr_datetime}_{title + '_std_' + str(execution_num)} ({function})")
    plt.savefig(path)

def save_avg_execution_metrics(avg_fit, std_fit, n_iters, perc_converged, avg_best_ind, avg_best_iter, function):
    curr_datetime = datetime.now().strftime('%m_%d_%H_%M_%S')
    path = os.path.join(os.getcwd(),"data",f"execution_metrics_{curr_datetime}_({function}).txt")

    with open(path, "w") as file:
        file_txt = f"Avg Fitness {avg_fit} \nStd Fitness {std_fit} \nNum. of iterations {n_iters}\n"
        file_txt += f"Perc. converged {perc_converged} \nAvg. Best Individual: {avg_best_ind}" 
        file_txt += f"Avg. Best Individual Iter: {avg_best_iter}" 
        file.write(file_txt)