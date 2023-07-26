import matplotlib.pyplot as plt
import numpy as np
from functions import Functions


def plot(function, points_limit, lower_limit, upper_limit, f_name="<unnamed>"):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Make data.
    x_range = [np.random.uniform(lower_limit,upper_limit) for x in range(points_limit)]
    y_range = [np.random.uniform(lower_limit,upper_limit) for y in range(points_limit)]
    z_list = [function((zipped[0],zipped[1])) for zipped in zip(x_range,y_range)]

    ax.scatter3D(x_range,y_range,z_list,c=z_list, cmap='jet', alpha=1, s=z_list)
    plt.title(f'{f_name} function Scatterplot', size=18)
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    ax.set_zlabel('z-axis')
    plt.show()


if __name__=="__main__":
  points_limit = 1000
  lower_limit = -32
  upper_limit = 32
  
  ack = Functions.ackley
  plot(ack, points_limit, lower_limit, upper_limit, "Ackley")