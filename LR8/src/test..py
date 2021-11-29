import graph
import mst

from main import ALGOS

N_tests = 10

if __name__ == '__main__':
    g = graph.generate_graph(3, 3)
    for algo, name in ALGOS:
        res = algo(g)
        assert len(res) == 2, f'{name} - {res}, on {g.adj_list}'
    print('==' * 10)
    print('BASE TESTS PASSED')
    print('==' * 10)

    N, M = 5, 10

    print('==' * 10)
    print('RANDOMIZED TESTS: RUNNING')
    print('==' * 10)
    for i in range(N_tests):
        g = graph.generate_graph(N, M)

        results = [(algo(g), name) for algo, name in ALGOS]

        for res1, name1 in results:
            for res2,name2 in results:
                s1, s2 = set(res1), set(res2)
                assert len(res1) == len(res2), f'{name1} vs {name2}: {len(res1)} - {len(res2)}'
                assert set(res1) == set(res2), f'{name1} vs {name2}: {s1.difference(s2)} - {s2.difference(s1)}'
        print(f'#{i + 1} PASSED')
    print('==' * 10)
    print('PASSED')
    print('==' * 10)

