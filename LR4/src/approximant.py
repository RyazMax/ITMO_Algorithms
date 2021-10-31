import numpy as np


class Approximant:
    def __init__(self, afunc, optimizer):
        self.afunc = afunc
        self.optimizer = optimizer
        self.params = ()

    def fit(self, X, y, initial, eps=1e-5):
        self.params, f_calls, iters = self.optimizer(
            lambda params: np.sum((self.afunc(X, params) - y) ** 2),
            args={
                'residuals': lambda params: self.afunc(X, params) - y,
            },
            initial=initial,
            eps=eps
        )
        print(f'Conv to {self.params} with calls: {f_calls} and iters: {iters}')

    def predict(self, X):
        return self.afunc(X, self.params)

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
    def function(X, params):
        if len(params) != 4:
            raise ValueError("Rational Approximant expects 4 params")
        a, b, c, d = params
        return (a * X + b) / (X ** 2 + c * X + d)

    def __init__(self, optimizer):
        super().__init__(afunc=RationalApproximant.function,
                     optimizer=optimizer)
