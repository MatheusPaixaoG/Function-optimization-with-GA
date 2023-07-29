from enum import Enum
#import random
#PARAMETERS

RUN = {
    # "random_seed": "jooj",
    "max_iterations": 10000,
    "population_size": 30,
    "print_step": 100
}

PRT_SEL = {
    "number_of_parents": 2,
    "num_indiv_selected": 5
}

CROSSOVER = {
    "alpha": 0.4,
    "type": "simple",
    "chance": 0.9
}

MUTATION = {
    "prob": 0.4
}

SVV_SEL = {
    "offspring_size": 2
}

class Functions(Enum):
    ACKLEY = 0
    RASTRIGIN = 1
    SCHWEFEL = 2
    ROSENBROCK = 3

FUNCTION = {
    "current_function": Functions.ACKLEY,
    "global_min": 0,
    "f_lo": -32.768,
    "f_hi": 32.768
}


""" # Max Iterations
max_iterations = 100

# Population Size
population_size = 10

# Selection of parents
number_of_parents = 2
num_indiv_selected = 5

#Crossover
cut_point = 2
alpha = 0.4
cross_type = "complete"
crossover_chance = 0.9

#Mutation
mutation_prob = 0.7

#Survivor Selection
offspring_size = 2

#Functions: ranges and global minimum
global_min = 4

# default => Ackley range
f_lo, f_hi = -32.768, 32.768 """
#random.seed(RUN["random_seed"])

def set_function(function_name):
    if(function_name == "ackley"):
        ackley()
    elif(function_name == "schwefel"):
        schwefel()
    elif(function_name == "rastrigin"):
        rastrigin()
    elif(function_name == "rosenbrock"):
        rosenbrock()
    else:
        raise Exception("Invalid function name")

def ackley():
    FUNCTION["f_lo"], FUNCTION["f_hi"] = -32.768, 32.768
    FUNCTION["current_function"] = Functions.ACKLEY

def schwefel():
    FUNCTION["f_lo"], FUNCTION["f_hi"] = -500, 500
    FUNCTION["current_function"] = Functions.SCHWEFEL

def rastrigin():
    FUNCTION["f_lo"], FUNCTION["f_hi"] = -5.12, 5.12
    FUNCTION["current_function"] = Functions.RASTRIGIN

def rosenbrock():
    FUNCTION["f_lo"], FUNCTION["f_hi"] = -5, 10
    FUNCTION["current_function"] = Functions.ROSENBROCK
