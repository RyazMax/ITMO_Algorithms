import random

import numpy as np
import matplotlib.pyplot as plt
import func
import approximant
import optimizer

APPROXIMANTS_CLS = [
    (approximant.LinearApproximant, 'LinearApproximant', [[0, 0], [1.5, 1.5]]),
    (approximant.RationalApproximant, 'RationalApproximant', [[-0.5, -1], [1, 1]]),
]
OPTIMIZERS = [
    (optimizer.brute_force2d, 'Brute-force method'),
    (optimizer.coordinate_descent, 'Gaus method'),
    (optimizer.nelder_mead, 'Nelder-Mead method'),
]
EPS = 1e-3

if __name__ == '__main__':
    random.seed(322)
    np.random.seed(322)
    x = np.arange(0, 100, dtype=float) / 100
    line, alpha, beta = func.random_line()
    noisy_line = func.additional_noise(line)
    y = line(x)
    y_noisy = noisy_line(x)
    print(f'Alpha: {alpha}, Beta: {beta}')

    for approximant_cls, approx_name, interval in APPROXIMANTS_CLS:
        for opt, opt_name in OPTIMIZERS:
            approx = approximant_cls(opt)
            approx.fit(x, y_noisy, interval=interval, eps=EPS)
            y_approx = approx(x)
            plt.plot(x, y_approx, label=opt_name)
            print(f"LOSS: {np.sum((y_approx - y_noisy)**2)}")

        plt.plot(x, y, label='Original line')
        print(f"ORIGINAL LOSS: {np.sum((y - y_noisy) ** 2)}")
        plt.plot(x, y_noisy, 'o', label='Noisy data')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(approx_name)
        plt.legend()
        plt.show()