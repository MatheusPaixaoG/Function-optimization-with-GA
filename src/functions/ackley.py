import numpy as np
import matplotlib.pyplot as plt

def ackley(point, a=20, b=0.2, c=2*np.pi):
    # point is the vector/tuple of coordinates
    d = len(point) 
    # First, we calculate the summation terms
    fst_sum = sum(np.multiply(point, point))
    snd_sum = sum(np.cos(np.multiply(c, point)))
    # Then, we calculate each term that uses a summation
    srqt_term = -a * np.exp(-b * np.sqrt(fst_sum/d))
    cos_term = -np.exp(snd_sum/d)
    # Now, we can calculate the result of the function
    result = srqt_term + cos_term + a + np.exp(1)
    return result

def plot_ackley_3d(points_limit, lower_limit, upper_limit):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Make data.
    x_range = [np.random.uniform(lower_limit,upper_limit) for x in range(points_limit)]
    y_range = [np.random.uniform(lower_limit,upper_limit) for y in range(points_limit)]
    z_list = [ackley((zipped[0],zipped[1])) for zipped in zip(x_range,y_range)]

    ax.scatter3D(x_range,y_range,z_list,c=z_list, cmap='jet', alpha=1, s=z_list)
    plt.title('Ackley function Scatterplot', size=18)
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    ax.set_zlabel('z-axis')
    plt.show()

if __name__=="__main__":
  points_limit = 1000
  lower_limit = -32
  upper_limit = 32
  #This would be the input for the Function
  plot_ackley_3d(points_limit, lower_limit, upper_limit) 