import os
import params

from evolutive.EE_utils import run_ee
from genetic.GA_utils import run_ga

if __name__ == "__main__":

    data_path = os.path.join(os.getcwd(),"data")

    if (not os.path.exists(data_path)):
        os.mkdir(data_path)

    params.set_function("ackley")
    points_limit = 50000

    #run_ga(2)
    run_ee()
