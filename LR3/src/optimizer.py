import typing as tp
import math
import scipy.optimize as scipy
import numpy as np


def grad_descent(func, args, initial, eps):
    lr = 1e-3
    grad = args.get('grad')
    a_cur, b_cur = initial
    a_prev, b_prev = a_cur - 1, b_cur - 1

    f_calls, iters = 0, 0
    while (a_prev - a_cur) ** 2 + (b_prev - b_cur)**2 > eps * eps:
        delta_a, delta_b = grad(a_cur, b_cur)
        a_prev, b_prev = a_cur, b_cur
        a_cur -= lr * delta_a
        b_cur -= lr * delta_b

        f_calls += 1
        iters += 1

    return (a_cur, b_cur), f_calls, iters


def conj_grad(func, args, initial, eps):
    res = scipy.minimize(fun=lambda x: func(x[0], x[1]), x0=initial, method='CG', tol=eps)
    return res.x, res.nfev, res.nit


def newton(func, args, initial, eps):
    res = scipy.minimize(fun=lambda x: func(x[0], x[1]), x0=initial, method='BFGS', tol=eps)
    return res.x, res.nfev, res.nit


def lma(func, args, initial, eps):
    func = args.get('residuals')
    res = scipy.least_squares(lambda x: func(x[0], x[1]), initial, method='lm', xtol=eps, ftol=eps)
    return res.x, res.nfev, res.nfev
