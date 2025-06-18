import unittest
import numpy as np
import networkx as nx
from src.clustering import calculate_adaptive_clustering_interval, spectral_clustering_with_temporal_kernel

class TestClustering(unittest.TestCase):

    def setUp(self):
        self.G = nx.karate_club_graph()
        self.spike_rates = np.random.rand(34)
        self.spike_times = np.random.rand(34) * 100

    def test_calculate_adaptive_clustering_interval(self):
        interval = calculate_adaptive_clustering_interval(self.G)
        self.assertIsInstance(interval, float)
        self.assertGreater(interval, 0)

    def test_spectral_clustering_with_temporal_kernel(self):
        k, labels = spectral_clustering_with_temporal_kernel(self.spike_rates, self.spike_times)
        self.assertIsInstance(k, int)
        self.assertEqual(len(labels), 34)

if __name__ == '__main__':
    unittest.main()
