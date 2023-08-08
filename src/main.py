import os
import params

from evolutive.EE_utils import run_ee
from genetic.GA_utils import run_ga

if __name__ == "__main__":

    data_path = os.path.join(os.getcwd(),"data")

    if (not os.path.exists(data_path)):
        os.mkdir(data_path)

    functions = ["ackley", "schwefel", "rastrigin", "rosenbrock"]
    params.set_function("ackley")
    points_limit = 50000

    # Run all functions
    for function in functions:
        params.set_function(function)
        run_ga(2, function)
        # run_ee(5, function)

    # run_ga(2)
    # # run_ee(1)