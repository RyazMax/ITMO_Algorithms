import networkx
import timeit
import random
import matplotlib.pyplot as plt
import algorithms
import numpy as np


def generate_map(n, m, obstacles):
    graph = networkx.grid_2d_graph(n, m)

    to_del = random.sample(graph.nodes(), obstacles)
    for node in to_del:
        graph.remove_node(node)

    inverse_graph = networkx.Graph()
    inverse_graph.add_nodes_from(to_del)

    return graph, inverse_graph


def draw_grid_graph(graph, color):
    pos = {(x, y): (y, -x) for x, y in graph.nodes()}
    networkx.draw(graph, pos=pos,
            node_color=color,
            with_labels=True,
            node_size=1000)


if __name__ == "__main__":
    plt.figure(figsize=(6, 6))

    random.seed(322)
    graph, obstacles = generate_map(10, 20, 40)

    N_EXPERIMENTS = 5
    times = []
    for _ in range(N_EXPERIMENTS):
        source, target = random.sample(graph.nodes(), 2)
        print(source, target)

        #result = algorithms.a_star(graph, source, target)
        times.append(timeit.timeit(lambda: algorithms.a_star(graph, source, target), number=1))
        #times.append(timeit.timeit(lambda: algorithms.dijkstra_networkx(graph, source, target), number=1))
        #draw_grid_graph(graph, 'lightblue')
        #draw_grid_graph(obstacles, 'gray')
        #result_graph = networkx.Graph()
        #result_graph.add_nodes_from(result)
        #draw_grid_graph(result_graph, 'red')

        #plt.show()
    times = np.array(times)
    print(times)
    print(f'Mean: {times.mean()}, STD: {times.std()}')