import math
import pandas as pd
import numpy as np

from dataclasses import dataclass
from collections import defaultdict

from sklearn.metrics.pairwise import haversine_distances
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.sparse import csr_matrix


# =====================================================
# MODELO
# =====================================================

@dataclass
class Edge:
    arc_id: str
    i: int
    j: int
    latitude: float
    longitude: float
    weight: float


# =====================================================
# CSV LOADER
# =====================================================

class CSVLoader:

    def load(self, path="edges.csv"):

        df = pd.read_csv(path)

        return [
            Edge(
                arc_id=row["arc_id"],
                i=row["i"],
                j=row["j"],
                latitude=row["latitude"],
                longitude=row["longitude"],
                weight=row["weight"]
            )
            for _, row in df.iterrows()
        ]


# =====================================================
# MATRIZ DE DISTÂNCIA (para p-median)
# =====================================================

class DistanceMatrixBuilder:

    def build(self, edges):

        coords = np.radians([
            [e.latitude, e.longitude]
            for e in edges
        ])

        dist = haversine_distances(coords) * 6371000

        return dist


# =====================================================
# P-MEDIANA CAPACITADO (SOBRE ARCOS)
# =====================================================

class CapacitatedPMedian:

    def solve(self, distance_matrix, demands, capacity, p):

        n = len(demands)

        medians = list(range(p))

        clusters = {m: [] for m in medians}
        loads = {m: 0 for m in medians}

        for i in range(n):

            best = None
            best_dist = math.inf

            for m in medians:

                if loads[m] + demands[i] <= capacity:

                    if distance_matrix[i][m] < best_dist:

                        best_dist = distance_matrix[i][m]
                        best = m

            if best is not None:
                clusters[best].append(i)
                loads[best] += demands[i]

        return clusters, loads


# =====================================================
# GRAFO (i,j)
# =====================================================

class GraphBuilder:

    def build(self, edges):

        index = {e.arc_id: i for i, e in enumerate(edges)}

        n = len(edges)
        mat = np.zeros((n, n))

        for a in edges:
            for b in edges:

                if a.arc_id == b.arc_id:
                    continue

                # ADJACÊNCIA POR VÉRTICE
                if len({a.i, a.j} & {b.i, b.j}) > 0:
                    mat[index[a.arc_id]][index[b.arc_id]] = 1

        return mat, index


# =====================================================
# CONECTIVIDADE REAL
# =====================================================

class ConnectivityService:

    def is_connected(self, graph, nodes):

        sub = graph[np.ix_(nodes, nodes)]

        ncomp, _ = connected_components(
            csr_matrix(sub),
            directed=False
        )

        return ncomp == 1


# =====================================================
# SHORTEST PATH
# =====================================================

class ShortestPathService:

    def compute(self, graph):

        return shortest_path(
            csr_matrix(graph),
            directed=False
        )


# =====================================================
# PIPELINE FINAL CORRIGIDO
# =====================================================

if __name__ == "__main__":

    loader = CSVLoader()
    edges = loader.load("edges.csv")

    demands = np.array([e.weight for e in edges])

    # ----------------------------
    # DISTÂNCIA (p-median)
    # ----------------------------

    dist_builder = DistanceMatrixBuilder()
    dist_matrix = dist_builder.build(edges)

    capacity = 90
    p = math.ceil(demands.sum() / capacity)

    solver = CapacitatedPMedian()
    clusters, loads = solver.solve(dist_matrix, demands, capacity, p)

    # ----------------------------
    # GRAFO REAL (i,j)
    # ----------------------------

    graph_builder = GraphBuilder()
    graph, index = graph_builder.build(edges)

    # ----------------------------
    # CONECTIVIDADE
    # ----------------------------

    checker = ConnectivityService()

    print("\n=== CONECTIVIDADE ===\n")

    fixed_clusters = {}

    for k, nodes in clusters.items():

        if checker.is_connected(graph, nodes):

            fixed_clusters[k] = nodes

        else:
            print(f"Cluster {k} NÃO conexo (precisa ajuste)")

    # ----------------------------
    # SHORTEST PATH
    # ----------------------------

    sp = ShortestPathService()
    dist = sp.compute(graph)

    print("\n=== RESULTADO FINAL ===\n")

    for k, nodes in fixed_clusters.items():

        print(f"\nCLUSTER {k}")
        print("ARESTAS:", [edges[i].arc_id for i in nodes])
        print("CARGA:", loads[k])

    print("\nMATRIZ MENOR CAMINHO:\n")
    print(dist)