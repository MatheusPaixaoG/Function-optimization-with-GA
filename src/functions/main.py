import matplotlib.pyplot as plt
import numpy as np
from functions import Ackley, Rastrigin, Schwefel, Rosenbrock

if __name__=="__main__":
    points_limit = 50000

    ack_low, ack_up = -32.768, 32.768
    ack = Ackley(points_limit, ack_low, ack_up)
    
    rast_low, rast_up = -5.12, 5.12
    rast = Rastrigin(points_limit, rast_low, rast_up)

    schw_low, schw_up = -500, 500
    schw = Schwefel(points_limit, schw_low, schw_up)
    
    ros_low, ros_up = -5, 10
    ros = Rosenbrock(points_limit, ros_low, ros_up)

    ack.plot()
    rast.plot()
    schw.plot()
    ros.plot()
