import math

import networkx as nx
import numpy as np

from decentralizepy.graphs.Graph import Graph


class DEquiStatic(Graph):
    """
    D-EquiStatic graph.
    """

    def __init__(self, n_procs, seed=0, eps=None, p=None, M=None):
        """A function that generates static topology for directed graphs satisfying
            Pr( ||Proj(W)||_2 < eps ) >= 1 - p
        Args:
            n: number of nodes
            seed: an integer used as the random seed
            eps: the upper bound of l2 norm
            p: the probability that the l2 norm is bigger than eps
            M: communication cost. If M is not given, M is calculated from eps and p.
        Returns:
            K: a numpy array that specifies the communication topology.
            As: a sequence of basis index
        """
        super().__init__(n_procs)

        if not M:
            M = int(8 * math.log(2 * n_procs / p) / 3 / eps**2)
        # generating M graphs
        np.random.seed(seed)
        As = np.random.choice(np.arange(1, n_procs), size=M, replace=True)
        Ws = np.zeros((n_procs, n_procs))
        for a in As:
            W = np.zeros((n_procs, n_procs))
            for i in range(1, n_procs + 1):
                j =  (i + a) % n_procs
                if j == 0: j = n_procs
                W[i-1, j-1] = (n_procs - 1) / n_procs
                W[i-1, i-1] = 1 / n_procs
            Ws += W

        K = Ws / M
        G = nx.from_numpy_array(K, create_using=nx.DiGraph)

        for edge in list(G.edges):
            node1 = edge[0]
            node2 = edge[1]
            if node1 == node2:
                continue
            self.adj_list[node1].add(node2)


class UEquiStatic(Graph):

    def __init__(self, n_procs, seed=0, eps=None, p=None, M=None):
        """A function that generates static topology for undirected graphs satisfying
            Pr( ||Proj(W)||_2 < eps ) >= 1 - p
        Args:
            n_procs: number of nodes
            seed: an integer used as the random seed
            eps: the upper bound of l2 norm
            p: the probability that the l2 norm is bigger than eps
            M: conmunnication cost. If M is not given, M is calculated from eps and p.
        Returns:
            K: a numpy array that specifies the communication topology.
            As: a sequence of basis index
        """
        super().__init__(n_procs)

        if M == None:
            M = int(8 * math.log(2 * n_procs / p) / 3 / eps ** 2)
        # generating M graphs
        np.random.seed(seed)
        As = np.random.choice(np.arange(1, n_procs), size=M, replace=True)
        Ws = np.zeros((n_procs, n_procs))
        for a in As:
            W = np.zeros((n_procs, n_procs))
            for i in range(1, n_procs + 1):
                j = (i + a) % n_procs
                if j == 0: j = n_procs
                W[i - 1, j - 1] = (n_procs - 1) / n_procs
                W[i - 1, i - 1] = 1 / n_procs
            Ws += W + W.T

        K = Ws / M / 2
        G = nx.from_numpy_array(K, create_using=nx.DiGraph)

        for edge in list(G.edges):
            node1 = edge[0]
            node2 = edge[1]
            if node1 == node2:
                continue
            self.adj_list[node1].add(node2)
            self.adj_list[node2].add(node1)
