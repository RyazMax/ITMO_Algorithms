import mst
import graph

import matplotlib.pyplot as plt
import numpy as np
import timeit

ALGOS = [
    #(mst.prim_base, r'Prim ~ $O(nm)$'),
    (mst.kruskal_base, r'Kruskal ~ $O(mlogm + n^2)$'),
    (mst.prim_dense, r'Prim ~ $O(n^2)$'),
    (mst.prim_sparse, r'Prim ~ $O(mlogn)$'),
    (mst.kruskal_dsu, r'Kruskal ~ $O(mlogm)$'),
]

EXPERIMENT_NUM = 7


def dense_graph_params(n_start, n_stop, step, ro=1, log=False):
    while n_start < n_stop:
        v = int(ro * n_start * (n_start - 1) / 2)

        if log:
            print(f'PARAMS: N:{n_start}, M:{v}')
        yield n_start, n_start, v
        n_start += step


if __name__ == '__main__':
    N, M = 200, 500
    N_max, STEP = 2500, 50

    x_label = 'N' # 'M'
    xs = []
    results = {name: [] for _, name in ALGOS}
    for x, n, m in dense_graph_params(N, N_max, STEP, log=True, ro=0.01):
        g = graph.generate_graph(n, m)
        xs.append(x)

        for algo, name in ALGOS:
            results[name].append(timeit.timeit(lambda: algo(g), number=EXPERIMENT_NUM) / EXPERIMENT_NUM)

    for name, vals in results.items():
        print(name, vals)
        plt.plot(xs, vals, label=name)

    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel('t(s)')
    plt.show()