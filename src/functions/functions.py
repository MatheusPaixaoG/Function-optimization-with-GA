import math
import numpy as np

class Functions:
    def __init__(self, name):
        self.name = name

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

    def rastrigin(x):
        return 10 * len(x) + sum([(i ** 2) - 10 * math.cos(2 * math.pi * i) for i in x])

    def schwefel(x):
        return 418.9829 * len(x) - sum([i * math.sin(math.sqrt(abs(i))) for i in x])

    def rosenbrock(x):
        return sum([100 * (x[x.index(i)+1] - i ** 2) ** 2 + (i - 1) ** 2 for i in x])