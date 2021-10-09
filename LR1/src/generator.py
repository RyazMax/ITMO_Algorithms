import numpy as np


def vector(n: int) -> np.ndarray:
    return np.random.rand(n)


def matrixes(n: int):
    return np.random.rand(n, n), np.random.rand(n, n)
