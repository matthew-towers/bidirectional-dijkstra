import networkx as nx
import queue
import random
import unittest
from collections import defaultdict


def bidirectional_dijkstra(g, s, t, weight):
    """
    Args:
        g: undirected networkx graph
        s: start vertex in g
        t: target vertex in g, distinct from s
        weight: weight function for g, taking two nodes and returning a number

    Returns:
        The length of the shortest path from s to t in g computed using
        bidirectional Dijkstra.
    """

    df = defaultdict(lambda: float("inf"))  # df[v] = forward approximation of d(s, v)
    df[s] = 0
    db = defaultdict(lambda: float("inf"))  # db[v] = backward approximation of d(t, v)
    db[t] = 0

    fq = queue.PriorityQueue()  # queue of vertices to explore in foward search
    fq.put((0, s))  # queue elements have the form (priority, node)
    bq = queue.PriorityQueue()  # queue of vertices to explore in backward search
    bq.put((0, t))

    mu = float("inf")  # length of best path yet seen

    sf = set()  # nodes processed in the forward search
    sb = set()  # nodes processed in the backward search

    while (not fq.empty()) and (not bq.empty()):
        u = fq.get()[1]  # get is a "pop"
        v = bq.get()[1]
        sf.add(u)
        sb.add(v)

        for x in g.adj[u]:
            # relax u-x
            if (x not in sf) and df[x] > df[u] + weight(u, x):
                df[x] = df[u] + weight(u, x)
                fq.put((df[x], x))

            # check for a path s --- u - x --- t, and update mu
            if (x in sb) and (df[u] + weight(u, x) + db[x] < mu):
                mu = df[u] + weight(u, x) + db[x]

        for x in g.adj[v]:
            # relax v-x
            if (x not in sb) and db[x] > db[v] + weight(v, x):
                db[x] = db[v] + weight(v, x)
                bq.put((db[x], x))

            # check for a path t --- v - x --- s, and update mu
            if (x in sf) and (db[v] + weight(v, x) + df[x] < mu):
                mu = db[v] + weight(v, x) + df[x]

        # check the termination condition
        if df[u] + db[v] >= mu:
            return mu
    print("something badly wrong happened.")


def test_bidirectional_dijkstra(g, s, t, weight):
    """
    Args:
        g: undirected networkx graph
        s: node in g
        t: node in g, distinct from s
        weight: function taking two nodes and returning a number

    Returns:
        True if the networkx shortest_path_length function and the bidirectional
        Dijkstra implementation here agree on the length of the shortest path
        from s to t in g, otherwise False.
    """
    nxDistance = nx.shortest_path_length(g, s, t, lambda u, v, _: weight(u, v))
    myDistance = bidirectional_dijkstra(g, s, t, weight)
    return nxDistance == myDistance


##############
# Test cases #
##############


class BidirectionalDijkstraTests(unittest.TestCase):

    def test_diamond(self):
        #  /2\
        # 1  4
        #  \3/
        g = nx.Graph()  # nx.Graph is undirected
        g.add_nodes_from([1, 2, 3, 4])
        g.add_edge(1, 2)
        g.add_edge(2, 4)
        g.add_edge(1, 3)
        g.add_edge(3, 4)
        weights = {(1, 2): 1, (2, 4): 2, (1, 3): 1, (3, 4): 1}

        def w(x, y):
            return weights[(x, y)] if (x, y) in weights else weights[(y, x)]

        self.assertTrue(test_bidirectional_dijkstra(g, 1, 4, w))

    def test_stackoverflow_graph(self):
        # Example from https://stackoverflow.com/questions/68768498
        #
        #     1   1   1   1
        #   E - D - C - B - A
        # 1 |               | 8
        #   F - G - H - I - J
        #     1   1   1   1
        #
        # The true shortest path from A to J has length 8, and the asker says
        # that a Java implementation of bidirectional Dijkstra gives the wrong
        # result.
        h = nx.Graph()
        h.add_nodes_from(list("ABCDEFGHIJ"))
        weights = {
            ("E", "D"): 1,
            ("D", "C"): 1,
            ("C", "B"): 1,
            ("B", "A"): 1,
            ("A", "J"): 8,
            ("E", "F"): 1,
            ("F", "G"): 1,
            ("G", "H"): 1,
            ("H", "I"): 1,
            ("I", "J"): 1,
        }
        for e in weights.keys():
            h.add_edge(e[0], e[1])

        def w(x, y):
            return weights[(x, y)] if (x, y) in weights else weights[(y, x)]

        self.assertTrue(test_bidirectional_dijkstra(h, "A", "J", w))

    def test_6006_graph(self):
        # Example from 6.006 recitation 16
        h = nx.Graph()
        h.add_nodes_from(["s", "u", "w", "up", "t"])
        h.add_edges_from([("s", "u"), ("u", "up"), ("up", "t"), ("s", "w"), ("w", "t")])
        weights = {
            ("s", "u"): 3,
            ("u", "up"): 3,
            ("up", "t"): 3,
            ("s", "w"): 5,
            ("w", "t"): 5,
        }

        def w(x, y):
            return weights[(x, y)] if (x, y) in weights else weights[(y, x)]

        self.assertTrue(test_bidirectional_dijkstra(h, "s", "t", w))
        self.assertTrue(test_bidirectional_dijkstra(h, "w", "up", w))
        self.assertTrue(test_bidirectional_dijkstra(h, "w", "u", w))

    def test_random_graphs(self):
        n_nodes = 200
        max_weight = 30
        n_tests = 50
        k_gnm = nx.gnm_random_graph(n_nodes, int(n_nodes * (n_nodes - 1) * 0.5 * 0.02))
        # gnm random graph is chosen uniformly at random from all graphs with n
        # vxs and m edges. Max number of edges is (1/2)*100*99
        k_er = nx.erdos_renyi_graph(n_nodes, 0.1)
        # erdos_renyi_graph(n_nodes, p) generates edges with probability p
        k_ba = nx.barabasi_albert_graph(n_nodes, 2)

        for k in [k_gnm, k_er, k_ba]:
            weights = {}

            def wt(x, y):
                # return weight of x-y, if set, otherwise generate and store a
                # random value
                if (x, y) in weights:
                    return weights[(x, y)]
                elif (y, x) in weights:
                    return weights[(y, x)]
                else:
                    weights[(x, y)] = random.randint(1, max_weight)
                    return weights[(x, y)]

            for _ in range(n_tests):
                s = random.choice(list(k.nodes))
                t = random.choice(list(nx.node_connected_component(k, s)))
                if s != t:
                    # print("cc size", len(nx.node_connected_component(k, s)))
                    self.assertTrue(test_bidirectional_dijkstra(k, s, t, wt))


if __name__ == "__main__":
    unittest.main()
