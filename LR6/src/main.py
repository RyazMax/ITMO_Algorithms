import queue
import random
from collections import Counter, deque
from pyvis.network import Network
import networkx

import algorithms
import timeit

def generate_graph(vertex_count, edges_count):
    vertex = list(range(vertex_count))
    edges = random.sample([(i, j, random.randint(5, 20)) for i in range(vertex_count) for j in range(i + 1, vertex_count)], edges_count)
    adj_matrix = [[0] * vertex_count for _ in range(vertex_count)]
    adj_list = [[] for _ in range(vertex_count)]
    for i, j, w in edges:
        adj_matrix[i][j] = w
        adj_matrix[j][i] = w

        adj_list[i].append((j, w))
        adj_list[j].append((i, w))

    return vertex, edges, adj_matrix, adj_list


def build_route(start, parents):
    route = [start]
    while parents[start] != -1:
        start = parents[start]
        route.append(start)
    return route[::-1]


def show_graph(vertex, edges, components=None, title='example.html'):
    net = Network(height='700px', width='1024px', notebook=True)

    for node in vertex:
        net.add_node(node, shape='circle')
    for s, t, w in edges:
        net.add_edge(s, t, title=str(w))

    net.show(title)


if __name__ == "__main__":
    random.seed(300)

    vertex, edges, adj_matrix, adj_list = generate_graph(100, 750)

    #show_graph(vertex, edges, title='all_graph')


    start, end = random.randint(0, len(adj_list) - 1), random.randint(0, len(adj_list) - 1)

    NUM_EXPERIMENTS = 20
    print(timeit.timeit(lambda: algorithms.dijkstra_algorithm(adj_list, start), number=NUM_EXPERIMENTS) / NUM_EXPERIMENTS)


    print(timeit.timeit(lambda: algorithms.bellman_ford(edges, len(vertex), start), number=NUM_EXPERIMENTS) / NUM_EXPERIMENTS)
    #route = build_route(parents=parents2, start=end)

    #assert dist1[end] == dist2[end]
    #assert parents1 == parents2

    #route_verts = [0 if i in route else 1 for i in vertex]
    #show_graph(vertex, edges, route_verts, 'route.html')