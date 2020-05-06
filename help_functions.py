import numpy as np

def one_hot(y, depth):
    one_hot_y = np.zeros(shape=(y.size, depth))
    one_hot_y[np.arange(y.size), y] = 1
    return one_hot_y
