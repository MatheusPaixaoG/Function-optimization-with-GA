import math
import matplotlib.pyplot as plt
import numpy as np

class Functions:
    def __init__(self, points_limit, lower_limit, upper_limit):
        self.points_limit = points_limit
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
    
    def plot(self, function):
        fig = plt.figure(figsize=(40,40))
        ax = fig.add_subplot(projection='3d')

        # Make data.
        y_range = [np.random.uniform(self.lower_limit,self.upper_limit) for y in range(self.points_limit)]
        x_range = [np.random.uniform(self.lower_limit,self.upper_limit) for x in range(self.points_limit)]
        z_list = [function((zipped[0],zipped[1])) for zipped in zip(x_range,y_range)]

        ax.scatter3D(x_range,y_range,z_list,c=z_list, cmap='jet', alpha=1, s=50)
        plt.title("WOW Look at Him!", size=18)
        plt.xlabel('x-axis')
        plt.ylabel('y-axis')
        ax.set_zlabel('z-axis')
        plt.show()

class Ackley(Functions):
    def __init__(self, points_limit, lower_limit, upper_limit):
        super().__init__(points_limit, lower_limit, upper_limit)
    
    def ackley(self, point, a=20, b=0.2, c=2*np.pi):
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
    
    def calculate(self, point):
        return self.ackley(point)
    
    def plot(self):
        return super().plot(super().ackley)


class Rastrigin(Functions):
    def __init__(self, points_limit, lower_limit, upper_limit):
        super().__init__(points_limit, lower_limit, upper_limit)
    
    def rastrigin(self, point):
        return 10 * len(point) + sum([(i ** 2) - 10 * math.cos(2 * math.pi * i) for i in point])
    
    def calculate(self, point):
        return self.rastrigin(point)
    
    def plot(self):
        return super().plot(super().rastrigin)
    

class Schwefel(Functions):
    def __init__(self, points_limit, lower_limit, upper_limit):
        super().__init__(points_limit, lower_limit, upper_limit)

    def schwefel(self, point):
        return 418.9829 * len(point) - sum([i * math.sin(math.sqrt(abs(i))) for i in point])
    
    def calculate(self, point):
        return self.schwefel(point)
    
    def plot(self):
        return super().plot(super().schwefel)
    

class Rosenbrock(Functions):
    def __init__(self, points_limit, lower_limit, upper_limit):
        super().__init__(points_limit, lower_limit, upper_limit)

    def rosenbrock(self, point):
        sum_list = []
        for i in range(len(point)-1):
            sum_list.append(100 * (point[i+1] - point[i] ** 2) ** 2 + (point[i] - 1) ** 2)
        return sum(sum_list)

    def calculate(self, point):
        return self.rosenbrock(point)
    
    def plot(self):
        return super().plot(super().rosenbrock)