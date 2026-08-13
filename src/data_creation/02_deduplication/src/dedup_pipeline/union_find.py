"""Small disjoint-set / union-find implementation for Stage A."""

parent = {}
rank = {}


def reset_union_find():
    parent.clear()
    rank.clear()


def uf_find(x):
    """Find the representative node with path compression."""
    if x not in parent:
        parent[x] = x
        rank[x] = 0
        return x

    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def uf_union(a, b):
    """Union two nodes by rank."""
    ra, rb = uf_find(a), uf_find(b)
    if ra == rb:
        return

    ra_rank, rb_rank = rank[ra], rank[rb]
    if ra_rank < rb_rank:
        parent[ra] = rb
    elif ra_rank > rb_rank:
        parent[rb] = ra
    else:
        parent[rb] = ra
        rank[ra] = ra_rank + 1
