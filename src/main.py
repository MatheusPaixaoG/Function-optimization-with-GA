import params
from genetic.GA_utils import run_ga

if __name__ == "__main__":
    params.set_function("ackley")
    points_limit = 50000

    run_ga()
