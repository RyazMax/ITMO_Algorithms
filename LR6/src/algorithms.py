import heapq
import networkx

MAX_PATH_LENGTH = 10 ** 10


def dijkstra_algorithm(adj_list, source):
    dist = [MAX_PATH_LENGTH] * len(adj_list)
    parent = [-1] * len(adj_list)
    dist[source] = 0
    queue = list(zip(dist, range(0, len(dist))))
    heapq.heapify(queue)

    while len(queue) > 0:
        c_distant, current = heapq.heappop(queue)
        if c_distant != dist[current]:
            continue
        for to, w in adj_list[current]:
            if dist[current] + w < dist[to]:
                dist[to] = dist[current] + w
                parent[to] = current
                heapq.heappush(queue, (dist[to], to))

    return dist, parent


def bellman_ford(edges, n, source, ):
    dist = [MAX_PATH_LENGTH] * n
    dist[source] = 0
    parent = [-1] * n

    changed = True
    while changed:
        changed = False
        for i, j, w in edges:
            if dist[i] < MAX_PATH_LENGTH or dist[j] < MAX_PATH_LENGTH:
                if dist[i] + w < dist[j]:
                    parent[j] = i
                    dist[j] = dist[i] + w
                    changed = True
                if dist[j] + w < dist[i]:
                    parent[i] = j
                    dist[i] = dist[j] + w
                    changed = True

    return dist, parent


def a_star(graph, source, target):
    return networkx.algorithms.shortest_paths.astar_path(graph, source=source, target=target)


def dijkstra_networkx(graph, source, target):
    return networkx.algorithms.shortest_paths.dijkstra_path(graph, source=source, target=target)