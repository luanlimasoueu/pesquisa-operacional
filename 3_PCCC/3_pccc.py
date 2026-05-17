import math
import pandas as pd
import numpy as np

from dataclasses import dataclass
from collections import defaultdict

from sklearn.metrics.pairwise import (
    haversine_distances
)

from scipy.sparse.csgraph import (
    connected_components
)

from scipy.sparse import csr_matrix

from scipy.sparse.csgraph import shortest_path


# =====================================================
# MODELO DA ARESTA
# =====================================================

@dataclass
class Edge:

    id: str

    latitude: float

    longitude: float

    demand: float


# =====================================================
# LEITURA DO CSV
# =====================================================

class CSVLoader:

    def load(self, path):

        df = pd.read_csv(path)

        edges = []

        for _, row in df.iterrows():

            edge = Edge(
                id=row["id"],
                latitude=row["latitude"],
                longitude=row["longitude"],
                demand=row["demanda"]
            )

            edges.append(edge)

        return edges


# =====================================================
# MATRIZ DE DISTÂNCIAS
# =====================================================

class DistanceMatrixBuilder:

    def build(self, edges):

        coords = np.array([
            [e.latitude, e.longitude]
            for e in edges
        ])

        # converte para radianos
        coords = np.radians(coords)

        # matriz haversine
        matrix = haversine_distances(coords)

        # converte para metros
        matrix *= 6371000

        return matrix


# =====================================================
# GCpMP SIMPLIFICADO
# =====================================================

class CapacitatedPMedian:

    def solve(
        self,
        distance_matrix,
        demands,
        capacity,
        p
    ):

        n = len(demands)

        # escolhe p medianas iniciais
        medians = list(range(p))

        clusters = {
            i: [] for i in medians
        }

        cluster_loads = {
            i: 0 for i in medians
        }

        # associa cada aresta
        for node in range(n):

            best_median = None
            best_distance = math.inf

            for median in medians:

                # verifica capacidade
                if (
                    cluster_loads[median]
                    + demands[node]
                    <= capacity
                ):

                    dist = (
                        distance_matrix[node][median]
                    )

                    if dist < best_distance:

                        best_distance = dist
                        best_median = median

            if best_median is not None:

                clusters[best_median].append(node)

                cluster_loads[best_median] += (
                    demands[node]
                )

        return clusters, cluster_loads


# =====================================================
# GRAFO DE CONECTIVIDADE
# =====================================================

class ConnectivityGraph:

    def build(
        self,
        edges,
        threshold=120
    ):

        n = len(edges)

        matrix = np.zeros((n, n))

        coords = np.array([
            [e.latitude, e.longitude]
            for e in edges
        ])

        coords = np.radians(coords)

        distances = haversine_distances(
            coords
        )

        distances *= 6371000

        # conecta arestas próximas
        for i in range(n):

            for j in range(n):

                if i != j:

                    if distances[i][j] <= threshold:

                        matrix[i][j] = distances[i][j]

        return matrix


# =====================================================
# TESTE DE CONECTIVIDADE
# =====================================================

class ConnectivityService:

    def is_connected(
        self,
        graph_matrix,
        cluster_nodes
    ):

        subgraph = graph_matrix[
            np.ix_(cluster_nodes, cluster_nodes)
        ]

        sparse = csr_matrix(subgraph)

        n_components, labels = (
            connected_components(
                csgraph=sparse,
                directed=False,
                return_labels=True
            )
        )

        return n_components == 1


# =====================================================
# CAMINHO MÍNIMO
# =====================================================

class ShortestPathService:

    def compute(
        self,
        graph_matrix
    ):

        distances = shortest_path(
            csgraph=graph_matrix,
            directed=False
        )

        return distances


# =====================================================
# RESULTADO FINAL
# =====================================================

class ClusterReporter:

    def show(
        self,
        clusters,
        cluster_loads,
        edges
    ):

        print("\n========== RESULTADO ==========\n")

        for cluster_id, nodes in clusters.items():

            print(f"CLUSTER {cluster_id}")

            edge_names = [
                edges[n].id
                for n in nodes
            ]

            print(
                "ARESTAS:",
                edge_names
            )

            print(
                "PESO TOTAL:",
                cluster_loads[cluster_id]
            )

            print()


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    # ---------------------------------
    # LEITURA CSV
    # ---------------------------------

    loader = CSVLoader()

    edges = loader.load(
        "edges.csv"
    )

    # ---------------------------------
    # MATRIZ DE DISTÂNCIAS
    # ---------------------------------

    matrix_builder = (
        DistanceMatrixBuilder()
    )

    distance_matrix = (
        matrix_builder.build(edges)
    )

    # ---------------------------------
    # DEMANDAS
    # ---------------------------------

    demands = np.array([
        e.demand for e in edges
    ])

    # ---------------------------------
    # CAPACIDADE
    # ---------------------------------

    capacity = 90

    # ---------------------------------
    # QUANTIDADE DE CLUSTERS
    # ---------------------------------

    p = math.ceil(
        demands.sum() / capacity
    )

    print("Quantidade clusters:", p)

    # ---------------------------------
    # PMEDIANAS CAPACITADO
    # ---------------------------------

    solver = CapacitatedPMedian()

    clusters, loads = solver.solve(
        distance_matrix,
        demands,
        capacity,
        p
    )

    # ---------------------------------
    # GRAFO
    # ---------------------------------

    graph_builder = (
        ConnectivityGraph()
    )

    graph_matrix = (
        graph_builder.build(edges)
    )

    # ---------------------------------
    # CONECTIVIDADE
    # ---------------------------------

    connectivity = (
        ConnectivityService()
    )

    print("\n========== CONECTIVIDADE ==========\n")

    for cluster_id, nodes in clusters.items():

        connected = (
            connectivity.is_connected(
                graph_matrix,
                nodes
            )
        )

        print(
            f"Cluster {cluster_id}:",
            connected
        )

    # ---------------------------------
    # CAMINHO MÍNIMO
    # ---------------------------------

    shortest = (
        ShortestPathService()
    )

    shortest_matrix = (
        shortest.compute(graph_matrix)
    )

    print("\n========== MATRIZ MENOR CAMINHO ==========\n")

    print(shortest_matrix)

    # ---------------------------------
    # RESULTADOS
    # ---------------------------------

    reporter = ClusterReporter()

    reporter.show(
        clusters,
        loads,
        edges
    )