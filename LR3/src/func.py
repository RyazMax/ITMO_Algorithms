import random
import numpy as np


def linear_ls_a(X, a, b):
    return a + b


def linear_ls_b(X, a, b):
    return a + b


def rational_ls_a(X, a, b):
    return a + b


def rational_ls_b(X, a, b):
    return a + b


def random_line():
    a = random.random()
    b = random.random()
    return lambda x: a * x + b, a, b


def additional_noise(func):
    return lambda x: func(x) + np.random.normal(size=len(x))
