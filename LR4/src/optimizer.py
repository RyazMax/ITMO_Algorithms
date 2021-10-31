import typing as tp
import math
import scipy.optimize as scipy
import numpy as np


def lma(func, args, initial, eps):
    func = args.get('residuals')
    res = scipy.least_squares(func, initial, method='lm', xtol=eps, ftol=eps)
    return res.x, res.nfev, res.nfev


def simulate_annealing(func, args, initial, eps):
    res = scipy.dual_annealing(func, bounds=list(zip([-4] * 4, [4] * 4)), x0=initial, no_local_search=True, maxiter=1000)
    return res.x, res.nfev, res.nit


def nelder_mead(func, args, initial, eps):
    res = scipy.minimize(func, initial, method='Nelder-Mead', tol=eps)
    return res.x, res.nfev, res.nit


def diff_evolution(func, args, initial, eps):
    res = scipy.differential_evolution(func, bounds=list(zip([-4] * 4, [4] * 4)), x0=initial, maxiter=1000)
    return res.x, res.nfev, res.nit
