import matplotlib.pyplot as plt
import numpy as np


class Function:
    def __init__(self, lower_limit, upper_limit):
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

    def plot(self, function, points_limit):
        fig = plt.figure(figsize=(40, 40))
        ax = fig.add_subplot(projection="3d")

        # Make data.
        y_range = [
            np.random.uniform(self.lower_limit, self.upper_limit)
            for y in range(points_limit)
        ]
        x_range = [
            np.random.uniform(self.lower_limit, self.upper_limit)
            for x in range(points_limit)
        ]
        z_list = [function((zipped[0], zipped[1])) for zipped in zip(x_range, y_range)]

        ax.scatter3D(x_range, y_range, z_list, c=z_list, cmap="jet", alpha=1, s=50)
        plt.title("WOW Look at Him!", size=18)
        plt.xlabel("x-axis")
        plt.ylabel("y-axis")
        ax.set_zlabel("z-axis")
        plt.show()
