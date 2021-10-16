import typing as tp
import math
import scipy.optimize as scipy
import numpy as np

def brute_force(func: tp.Callable[[float], float], interval: tp.Tuple[float, float],
                eps: float=1e-5) -> tp.Tuple[float, int, int]:
    """
    :param func: function to be minimized
    :param interval: [a, b] on which interval function to minimized
    :param eps: precision of minimization
    :return: x* sollution, N - how many times func was called, M - how many iteration was made
    """
    start, end = interval

    x_cur = start
    x_min = x_cur
    f_min = func(x_cur)

    f_calls, iters = 1, 1
    while x_cur <= end:
        x_cur += eps
        f = func(x_cur)
        if iters % 10000 == 0:
            print(x_cur, f)
        if f < f_min:
            f_min, x_min = f, x_cur
        f_calls += 1
        iters += 1
    return x_min + eps / 2, f_calls, iters


def dichotomy(func: tp.Callable[[float], float], interval: tp.Tuple[float, float],
                eps: float=1e-5) -> tp.Tuple[float, int, int]:
    """
    :param func: function to be minimized
    :param interval: [a, b] on which interval function to minimized
    :param eps: precision of minimization
    :return: x* sollution, N - how many times func was called, M - how many iteration was made
    """
    delta = eps / 23
    start, end = interval
    f_calls, iters = 0, 0
    while end - start > eps:
        mid = (start + end) / 2
        x1, x2 = mid - delta, mid + delta
        f1, f2 = func(x1), func(x2)
        f_calls += 2
        if f1 > f2:
            start = x1
        else:
            end = x2
        iters += 1
    return (start + end) / 2, f_calls, iters


def golden_section_search(func: tp.Callable[[float], float], interval: tp.Tuple[float, float],
                eps: float=1e-5) -> tp.Tuple[float, int, int]:
    """
    :param func: function to be minimized
    :param interval: [a, b] on which interval function to minimized
    :param eps: precision of minimization
    :return: x* sollution, N - how many times func was called, M - how many iteration was made
    """
    start, end = interval
    phi = (1 + math.sqrt(5)) / 2
    resphi = 2 - phi
    x1 = start + resphi * (end - start)
    x2 = end - resphi * (end - start)
    f1 = func(x1)
    f2 = func(x2)
    f_calls, iters = 2, 0
    while end - start > eps:
        if f1 < f2:
            end = x2
            x2 = x1
            f2 = f1
            x1 = start + resphi * (end - start)
            f1 = func(x1)
        else:
            start = x1
            x1 = x2
            f1 = f2
            x2 = end - resphi * (end - start)
            f2 = func(x2)
        f_calls += 1
        iters += 1
    return (start + end) / 2, f_calls, iters


def brute_force2d(
        func: tp.Callable[[float, float], float],
        interval: tp.Tuple[tp.Tuple[float, float], tp.Tuple[float, float]],
        eps: float=1e-5) -> tp.Tuple[tp.Tuple[float, float], int, int]:
    (h_start, v_start), (h_end, v_end) = interval
    f_calls, iters = 0, 0

    x_cur, y_cur = h_start, v_start
    x_min, y_min, f_min = x_cur, y_cur, float('inf')

    while x_cur <= h_end:
        y_cur = v_start
        while y_cur <= v_end:
            f = func(x_cur, y_cur)
            if f < f_min:
                x_min, y_min, f_min = x_cur, y_cur, f
            y_cur += eps
            f_calls += 1
            iters += 1

        x_cur += eps
    return (x_min + eps / 2, y_min + eps / 2), f_calls, iters

def coordinate_descent(
        func: tp.Callable[[float, float], float],
        interval: tp.Tuple[tp.Tuple[float, float], tp.Tuple[float, float]],
        eps: float=1e-5) -> tp.Tuple[tp.Tuple[float, float], int, int]:
    """
    Coordinate descent for 2d optimization
    :param func:
    :param interval:
    :param eps:
    :return:
    """

    (h_start, v_start), (h_end, v_end) = interval
    x_prev, y_prev = h_end, v_end
    x_cur, y_cur = h_start, v_start
    f_calls, iters = 0, 0
    while abs(x_prev - x_cur) > eps or abs(y_prev - y_cur) > eps:
        x_prev, y_prev = x_cur, y_cur
        x_cur, f_calls_x, iters_x = golden_section_search(lambda x: func(x, y_prev), [h_start, h_end])
        y_cur, f_calls_y, iters_y = golden_section_search(lambda y: func(x_cur, y), [v_start, v_end])

        f_calls += f_calls_x + f_calls_y
        iters += iters_x + iters_y

    return (x_cur, y_cur), f_calls, iters


def nelder_mead(
        func: tp.Callable[[float, float], float],
        interval: tp.Tuple[tp.Tuple[float, float], tp.Tuple[float, float]],
        eps: float=1e-5) -> tp.Tuple[tp.Tuple[float, float], int, int]:
    """
    Nelder-mead 2d optimization
    :param func:
    :param interval:
    :param eps:
    :return:
    """

    res = scipy.minimize(lambda x: func(x[0], x[1]), interval[0], method='Nelder-Mead', tol=eps)
    return res.x, res.nfev, res.nit