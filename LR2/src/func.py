import numpy as np
import random

def cubic(x):
    return x ** 3


def abs_shift(x):
    return np.abs(x - 0.2)


def xsin(x):
    return x * np.sin(1.0 / x)


def random_line():
    a = random.random()
    b = random.random()
    return lambda x: a * x + b, a, b


def additional_noise(func):
    return lambda x: func(x) + np.random.normal(size=len(x))
