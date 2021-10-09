import argparse

import numpy as np
import pandas as pd
import numpy.linalg as lg
from matplotlib import pyplot as plt


def linear_approx(x, y):
    def preprocess_x(x):
        return np.hstack((x[:, np.newaxis], np.ones((len(x), 1))))
    A = preprocess_x(x)
    w = lg.pinv(A) @ y[:, np.newaxis]
    return lambda x: preprocess_x(x) @ w


def const_approx(_, y):
    w = np.mean(y)
    return lambda x: np.ones_like(x) * w


def quad_approx(x, y):
    def preprocess_x(x):
        return np.hstack((x[:, np.newaxis] ** 2, x[:, np.newaxis], np.ones((len(x), 1))))

    A = preprocess_x(x)
    w = lg.pinv(A) @ y[:, np.newaxis]
    return lambda x: preprocess_x(x) @ w

def qubic_approx(x, y):
    def preprocess_x(x):
        return np.hstack((x[:, np.newaxis] ** 3,  x[:, np.newaxis] ** 2, x[:, np.newaxis], np.ones((len(x), 1))))
    A = preprocess_x(x)
    w = lg.pinv(A) @ y[:, np.newaxis]
    return lambda x: preprocess_x(x) @ w


def nlogn_approx(x, y):
    def preprocess_x(x):
        ones = np.ones((len(x), 1))
        x = x[:, np.newaxis]
        return np.hstack((x * np.log2(x), ones))
    A = preprocess_x(x)
    w = lg.pinv(A) @ y[:, np.newaxis]
    return lambda x: preprocess_x(x) @ w


APPROX = {
    'linear': linear_approx,
    'quad': quad_approx,
    'qubic': qubic_approx,
    'nlogn': nlogn_approx,
    'const': const_approx,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--approx', nargs='+', type=str, required=True)
    parser.add_argument('--title', type=str, required=True)

    args = parser.parse_args()

    df = pd.read_csv(args.input, header=None)
    x, y = df[0].to_numpy(), df[1].to_numpy()

    mask = y < 6
    x = x[mask]
    y = y[mask]

    plt.plot(x, y, label='experimental')

    for approx in args.approx:
        theoretical = APPROX[approx](x, y)
        y_t = theoretical(x)
        plt.plot(x, y_t, label=f'theoretical {approx}')
    plt.xlabel('N')
    plt.ylabel('time(s)')
    plt.title(args.title)
    plt.legend()
    plt.show()
