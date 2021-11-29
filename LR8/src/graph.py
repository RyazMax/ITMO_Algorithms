import typing
import random


class Edge:
    def __init__(self, start, target, w):
        self.start = start
        self.target = target
        self.w = w

    def inverse(self):
        return Edge(self.target, self.start, self.w)

    def __eq__(self, other):
        cond1 = self.start == other.start and self.target == other.target
        cond2 = self.start == other.target and self.target == other.start
        return (cond1 or cond2) and self.w == other.w

    def __repr__(self):
        return f'({self.start} -> {self.target}: {self.w})'

    def __hash__(self):
        return hash(tuple(sorted([self.start, self.target]) + [self.w]))


def generate_graph(n, m):
    vertex = list(range(n))
    edges = random.sample(
        [(i, j, random.randint(5, 20)) for i in range(n) for j in range(i + 1, n)], m)
    adj_matrix = [[0] * n for _ in range(n)]
    adj_list = [[] for _ in range(n)]
    for i, j, w in edges:
        adj_matrix[i][j] = w
        adj_matrix[j][i] = w

        adj_list[i].append((j, w))
        adj_list[j].append((i, w))

    return Graph([Edge(s, t, w) for s, t, w in edges], vertex, adj_list, adj_matrix)


class Graph:
    def __init__(self, edges, vertices, adj_list, adj_matrix):
        self.edges = edges
        self.vertices = vertices
        self.adj_list: typing.List[typing.List[typing.Tuple[int, int]]] = adj_list
        self.adj_matrix: typing.List[typing.List[int]] = adj_matrix

    def __repr__(self):
        return f'{self.edges}'

