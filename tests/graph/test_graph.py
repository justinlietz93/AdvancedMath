import unittest
import networkx as nx
from src.graph import (
    calculate_graph_metrics, 
    detect_communities, 
    calculate_graph_edit_distance, 
    calculate_pagerank,
    coarse_grain_graph,
    simulate_sparsity_evolution,
    calculate_path_score
)
import numpy as np

class TestGraph(unittest.TestCase):

    def setUp(self):
        self.G = nx.karate_club_graph()

    def test_calculate_graph_metrics(self):
        metrics = calculate_graph_metrics(self.G)
        self.assertIsInstance(metrics, dict)
        self.assertIn('num_nodes', metrics)
        self.assertEqual(metrics['num_nodes'], 34)

    def test_detect_communities(self):
        communities = detect_communities(self.G)
        self.assertIsInstance(communities, list)
        self.assertGreater(len(communities), 1)

    def test_calculate_graph_edit_distance(self):
        G2 = self.G.copy()
        G2.add_edge(0, 10)
        dist = calculate_graph_edit_distance(self.G, G2)
        self.assertGreater(dist, 0)

    def test_calculate_pagerank(self):
        pagerank = calculate_pagerank(self.G)
        self.assertIsInstance(pagerank, dict)
        self.assertEqual(len(pagerank), 34)

    def test_coarse_grain_graph(self):
        partitions = [[0, 1, 2], [3, 4, 5]]
        coarse_graph = coarse_grain_graph(self.G, partitions)
        self.assertEqual(coarse_graph.number_of_nodes(), 2)

    def test_simulate_sparsity_evolution(self):
        sparsity = simulate_sparsity_evolution(0.1, 0.1, 0.01, np.ones(10), 1.0, 10.0)
        self.assertEqual(len(sparsity), 11)

    def test_calculate_path_score(self):
        score = calculate_path_score(np.array([0.5, 0.5]), np.array([1, 1]), np.array([1, 1]), 1.0)
        self.assertIsInstance(score, float)

if __name__ == '__main__':
    unittest.main()
