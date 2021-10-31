import random

import numpy as np
import matplotlib.pyplot as plt
import func
import approximant
import optimizer

APPROXIMANTS_CLS = [
    (
        approximant.RationalApproximant,
        'RationalApproximant',
        [-1, 1, -2, 1],
    ),
]
OPTIMIZERS = [
     (optimizer.lma, 'Levenberg–Marquardt algorithm'),
     (optimizer.nelder_mead, 'Nelder-Mead'),
     (optimizer.simulate_annealing, 'Simulated annealing'),
     (optimizer.diff_evolution, 'Differential evolution'),
]
EPS = 1e-3

if __name__ == '__main__':
    random.seed(322)
    np.random.seed(322)
    x = 3 * np.arange(0, 1000, dtype=float) / 1000
    origin_func = func.strange_func()
    noisy_func = func.additional_noise(origin_func)
    y = origin_func(x)
    y_noisy = noisy_func(x)

    for approximant_cls, approx_name, initial in APPROXIMANTS_CLS:
        for opt, opt_name in OPTIMIZERS:
            approx = approximant_cls(opt)
            print(f'====={opt_name}=====')
            approx.fit(x, y_noisy, initial=initial, eps=EPS)
            y_approx = approx(x)
            plt.plot(x, y_approx, label=opt_name)
            print(f"LOSS: {np.sum((y_approx - y_noisy)**2)}\n")

        plt.plot(x, y, label='Original line')
        print(f"ORIGINAL LOSS: {np.sum((y - y_noisy) ** 2)}")
        plt.plot(x, y_noisy, '*', label='Noisy data')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(approx_name)
        plt.legend()
        plt.show()