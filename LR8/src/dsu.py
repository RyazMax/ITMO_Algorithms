import random


class DSU:
    def __init__(self, n):
        self.tree_id = list(range(n))

    def unite(self, a, b):
        a = self.get(a)
        b = self.get(b)
        if random.random() > 0.5:
            a, b = b, a
        if a != b:
            self.tree_id[a] =b

    def get(self, v):
        if v == self.tree_id[v]:
            return v
        else:
            self.tree_id[v] = self.get(self.tree_id[v])
            return self.tree_id[v]