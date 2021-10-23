import numpy as np


class Approximant:
    def __init__(self, afunc, grad, hess, optimizer):
        self.afunc = afunc
        self.gradient = grad
        self.hessian = hess
        self.optimizer = optimizer
        self.a, self.b = None, None

    def fit(self, X, y, initial, eps=1e-5):
        ls_d = lambda a, b: 2 * (self.afunc(X, a, b) - y)
        (a, b), f_calls, iters = self.optimizer(
            lambda a, b: np.sum((self.afunc(X, a, b) - y) ** 2),
            args={
                'grad': lambda a, b: np.sum(np.array(self.gradient(X, a, b)) * ls_d(a, b), axis=1),
                'hess': lambda a, b: self.hessian(X, a, b),
                'residuals': lambda a, b: self.afunc(X, a, b) - y,
            },
            initial=initial,
            eps=eps
        )
        print(f'Conv to {(a,b )} with calls: {f_calls} and iters: {iters}')
        self.a, self.b = a, b

    def predict(self, X):
        return self.afunc(X, self.a, self.b)

    def __call__(self, X):
        return self.predict(X)


class LinearApproximant(Approximant):

    @staticmethod
    def function(x, a, b):
        return a * x + b

    @staticmethod
    def gradient(x, a, b):
        return x, np.ones_like(x)

    @staticmethod
    def hessian(x, a, b):
        return x

    def __init__(self, optimizer):
        super().__init__(afunc=LinearApproximant.function,
                         grad=LinearApproximant.gradient,
                         hess=LinearApproximant.hessian,
                         optimizer=optimizer)


class RationalApproximant(Approximant):
    @staticmethod
    def function(x, a, b):
        return a / (1.0 + b * x)

    @staticmethod
    def gradient(x, a, b):
        return 1.0 / (1 + b * x), - (a * x) / (b * x + 1) ** 2

    @staticmethod
    def hessian(x):
        return x

    def __init__(self, optimizer):
        super().__init__(afunc=RationalApproximant.function,
                         grad=RationalApproximant.gradient,
                         hess=RationalApproximant.hessian,
                         optimizer=optimizer)