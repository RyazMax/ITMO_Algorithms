import queue
import random
from collections import Counter, deque
from pyvis.network import Network

def generate_graph(vertex_count, edges_count):
    vertex = list(range(vertex_count))
    edges = random.sample([(i, j) for i in range(vertex_count) for j in range(i + 1, vertex_count)], edges_count)
    adj_matrix = [[0] * vertex_count for _ in range(vertex_count)]
    adj_list = [[] for _ in range(vertex_count)]
    for i, j in edges:
        adj_matrix[i][j] = 1
        adj_matrix[j][i] = 1

        adj_list[i].append(j)
        adj_list[j].append(i)

    return vertex, edges, adj_matrix, adj_list


def dfs(i, vs, component, components, adj_list):
    components[i] = component
    for v in vs:
        if components[v] == -1:
            dfs(v, adj_list[v], component, components, adj_list)


def connected_components(adj_list):
    components = [-1] * len(adj_list)
    component = 0
    for i, vs in enumerate(adj_list):
        if components[i] == -1:
            dfs(i, vs, component, components, adj_list)
            component += 1
    return components


def bfs(adj_list, start):
    dist = [-1] * len(adj_list)
    parents = [-1] * len(adj_list)
    dist[start] = 0
    q = deque()
    q.append(start)
    while q:
        v = q.popleft()
        for to in adj_list[v]:
            if dist[to] < 0:
                dist[to] = dist[v] + 1
                parents[to] = v
                q.append(to)
    return dist, parents


def build_route(start, parents):
    route = [start]
    while parents[start] != -1:
        start = parents[start]
        route.append(start)
    return route[::-1]


def show_graph(vertex, edges, components, title='example.html'):
    net = Network(height='700px', width='1024px', notebook=True)

    for node in vertex:
        net.add_node(node, shape='circle', label=dist[node], group=components[node] + 1)
    net.add_edges(edges)

    net.show(title)


if __name__ == "__main__":
    random.seed(5)
    while True:
        vertex, edges, adj_matrix, adj_list = generate_graph(vertex_count=100, edges_count=200)
        components = connected_components(adj_list)
        counter = Counter(components)
        most_common = counter.most_common(3)
        if len(most_common) == 3 and most_common[1][1] + most_common[2][1] > 5:
            break

    print("MATRIX\n", adj_matrix[:3])
    print("LIST\n", adj_list[:3])
    print("COMPONENTS: ", max(components) + 1)

    #show_graph(vertex, edges, components)

    start, end = random.randint(0, len(adj_list) - 1), random.randint(0, len(adj_list) - 1)
    dist, parents = bfs(adj_list, start)
    route = build_route(parents=parents, start=end)
    print(f"FROM {start} to {end} takes: {dist[end]}, route: {route}")
    route_verts = [0 if i in route else 1 for i in vertex]

    show_graph(vertex, edges, route_verts, 'route.html')