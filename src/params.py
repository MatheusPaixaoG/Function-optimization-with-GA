from enum import Enum
#import random
#PARAMETERS

# GA PARAMETERS
RUN = {
    # "random_seed": "jooj",
    "max_iterations": 200_001,
    "population_size": 500,
    "print_step": 100
}

PRT_SEL = {
    "number_of_parents": 3,
    "num_indiv_selected": 5
}

CROSSOVER = {
    "alpha": 0.5,
    "type": "complete",
    "chance": 0.9,
    "offspring_size": 100
}

MUTATION = {
    "prob": 0.1,
    "forced_prob": 0.1,
    "force_mutate_tol": 1e-4,
    "force_mutate_it": 2000
}

# FUNCTION PARAMETERS
class Functions(Enum):
    ACKLEY = 0
    RASTRIGIN = 1
    SCHWEFEL = 2
    ROSENBROCK = 3

FUNCTION = {
    "current_function": Functions.ACKLEY,
    "global_min": 1e-7,
    "f_lo": -32.768,
    "f_hi": 32.768
}

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
