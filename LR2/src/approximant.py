import numpy as np


class Approximant:
    def __init__(self, afunc, optimizer):
        self.afunc = afunc
        self.optimizer = optimizer
        self.a, self.b = None, None

    def fit(self, X, y, interval=[[0, 0], [1, 1]], eps=1e-5):
        (a, b), f_calls, iters = self.optimizer(
            lambda a, b: np.sum((self.afunc(X, a, b) - y) ** 2),
            interval,
            eps=eps
        )
        print(f'Conv to {(a,b )} with calls: {f_calls} and iters: {iters}')
        self.a, self.b = a, b

    def predict(self, X):
        return self.afunc(X, self.a, self.b)

    def __call__(self, X):
        return self.predict(X)


class LinearApproximant(Approximant):
    def __init__(self, optimizer):
        super().__init__(lambda x, a, b: a * x + b, optimizer)


class RationalApproximant(Approximant):
    def __init__(self, optimizer):
        super().__init__(lambda x, a, b: a / (1 + b * x), optimizer)