import typing
import graph
import bintrees
import dsu

INF = 2 * 10 ** 9


# Trivial implementation O(nm) time complexity
def prim_base(g: graph.Graph) -> typing.List[graph.Edge]:
    adj_list = g.adj_list

    n = len(adj_list)

    mst_v: typing.Set[int] = {0}
    mst_e: typing.List[graph.Edge] = []

    for i in range(n - 1):
        min_e: typing.Optional[graph.Edge] = None
        for edge in g.edges:
            if edge.start in mst_v and edge.target not in mst_v:
                min_e = edge if min_e is None or min_e.w > edge.w else min_e
            edge = edge.inverse()
            if edge.start in mst_v and edge.target not in mst_v:
                min_e = edge if min_e is None or min_e.w > edge.w else min_e

        if min_e is not None:
            mst_e.append(min_e)
            mst_v.add(min_e.target)
        else:
            break
    return mst_e


# Prim algorithm for dense graphs with O(n^2) time and O(n) space complexity
# Use adjacency matrix for work
def prim_dense(g: graph.Graph) -> typing.List[graph.Edge]:
    adj_list = g.adj_list

    n = len(adj_list)

    mst_v: typing.Set[int] = set()
    mst_e: typing.List[graph.Edge] = []

    min_e: typing.List[typing.Tuple[int, int]] = [(INF, -1)] * n
    min_e[0] = (0, -1)
    for i in range(n):
        v = -1
        for j in range(n):
            if j not in mst_v and (v == -1 or min_e[j][0] < min_e[v][0]):
                v = j
        if min_e[v][0] == INF:
            break

        mst_v.add(v)
        if min_e[v][1] != -1:
            mst_e.append(graph.Edge(min_e[v][1], v, min_e[v][0]))

        for to, w in adj_list[v]:
            if w < min_e[to][0]:
                min_e[to] = (w, v)

    return mst_e


# Prim algorithm for sparse graphs with O(m log n) time complexity
def prim_sparse(g: graph.Graph) -> typing.List[typing.Tuple[int, int]]:
    adj_list = g.adj_list

    n = len(adj_list)

    mst_e: typing.List[graph.Edge] = []

    min_e: typing.List[typing.Tuple[int, int]] = [(INF, -1)] * n
    min_e[0] = (0, -1)
    q = bintrees.RBTree()
    q.insert((0, 0), None)
    for i in range(n):
        if q.is_empty():
            break
        w, v = q.min_key()
        q.remove((w, v))

        if min_e[v][1] != -1:
            mst_e.append(graph.Edge(min_e[v][1], v, min_e[v][0]))

        for to, w in adj_list[v]:
            if w < min_e[to][0]:
                if q.get((min_e[to][0], to)) is not None:
                    q.remove((min_e[to][0], to))
                min_e[to] = (w, v)
                q.insert((min_e[to][0], to), None)

    return mst_e


# Base Kruskal algorithm with O(m log n + n^2) time complexity
def kruskal_base(g: graph.Graph) -> typing.List[graph.Edge]:
    edges = sorted([(edge.w, edge.start, edge.target) for edge in g.edges])
    tree_id = list(range(len(g.vertices)))

    mst_e: typing.List[graph.Edge] = []
    for w, source, target in edges:
        if tree_id[source] != tree_id[target]:
            mst_e.append(graph.Edge(source, target, w))
            old_id, new_id = tree_id[target], tree_id[source]
            for i in range(len(tree_id)):
                if tree_id[i] == old_id:
                    tree_id[i] = new_id

    return mst_e


# Kruskal algorithm with  with O(m log n) time complexity
def kruskal_dsu(g: graph.Graph) -> typing.List[graph.Edge]:
    edges = sorted([(edge.w, edge.start, edge.target) for edge in g.edges])
    tree_id = dsu.DSU(len(g.vertices))

    mst_e: typing.List[graph.Edge] = []
    for w, source, target in edges:
        if tree_id.get(source) != tree_id.get(target):
            mst_e.append(graph.Edge(source, target, w))
            tree_id.unite(source, target)
    return mst_e
