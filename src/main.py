from functions.FitnessFunction import Functions
from genetic.Individual import Individual


def init_population(population_size, lower_limit, upper_limit, function_ID):
    population = [
        Individual(lower_limit, upper_limit, function_ID)
        for indiv in range(population_size)
    ]
    return population


def sort_by_fitness(population):
    population.sort(key=lambda x: x.fitness)


if __name__ == "__main__":
    points_limit = 50000

    ack_low, ack_up = -32.768, 32.768
    rast_low, rast_up = -5.12, 5.12
    schw_low, schw_up = -500, 500
    ros_low, ros_up = -5, 10

    population_size = 10
    population = init_population(population_size, ack_low, ack_up, Functions.ACKLEY)

    # for i in range(0, population_size):
    #     ind = population[i]
    #     print(f"{ind} {round(ind.calc_fitness(Functions.ACKLEY), 4)}")
    #     print(f"{ind} {round(ind.calc_fitness(Functions.RASTRIGIN), 4)}")
    #     print(f"{ind} {round(ind.calc_fitness(Functions.ROSENBROCK), 4)}")
    #     print(f"{ind} {round(ind.calc_fitness(Functions.SCHWEFEL), 4)}")

    sort_by_fitness(population)

    for indiv in population:
        print(f"{indiv}")
