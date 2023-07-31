from enum import Enum
#import random
#PARAMETERS

# GA PARAMETERS
RUN = {
    # "random_seed": "jooj",
    "max_iterations": 100_000,
    "population_size": 30,
    "print_step": 1000
}

PRT_SEL = {
    "number_of_parents": 2,
    "num_indiv_selected": 5
}

CROSSOVER = {
    "alpha": 0.4,
    "type": "normal",
    "chance": 0.9
}

MUTATION = {
    "prob": 0.4,
    "forced_prob": 1
}

SVV_SEL = {
    "offspring_size": 2
}

# FUNCTION PARAMETERS
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
