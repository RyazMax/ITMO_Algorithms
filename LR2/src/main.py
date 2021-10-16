import matplotlib.pyplot as plt
import numpy as np

import func
import optimizer

OPTIMIZERS = [
    (optimizer.brute_force, 'Brute force'),
    (optimizer.dichotomy, 'Dichotomy'),
    (optimizer.golden_section_search, 'Golden section search'),
]
FUNCS = [
    (func.cubic, (0.0, 1.0), 'x^3'),
    (func.abs_shift, (0.0, 1.0), '|x - 0.2|'),
    (func.xsin, (0.01, 1.0), 'x * sin(1 / x)'),
]
EPS = 1e-3
LINSPACE_NUM=500


if __name__ == "__main__":
    for func, interval, func_name in FUNCS:
        x_found = []
        for optimizer, name in OPTIMIZERS:
            x, f_calls, iters = optimizer(func, interval, eps=EPS)
            x_found.append((x, name))
            print(f'{name} optimizer, x: {x}, f: {func(x)}, f_calls: {f_calls}, iters: {iters}')
        x = np.linspace(interval[0], interval[1], num=LINSPACE_NUM)
        y = func(x)

        plt.plot(x, y)
        for x_opt, name in x_found:
            plt.plot(x_opt, func(x_opt), 'o', label=name)

        plt.legend()
        plt.title(func_name)
        plt.show()