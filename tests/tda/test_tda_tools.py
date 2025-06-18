import unittest
import numpy as np
from src.tda import construct_vietoris_rips, compute_persistent_homology, calculate_tda_metrics

class TestTDATools(unittest.TestCase):

    def setUp(self):
        # A simple point cloud: two clusters
        self.points1 = np.array([
            [0, 0], [1, 0], [0, 1], [1, 1],  # Cluster 1
            [5, 5], [6, 5], [5, 6], [6, 6]   # Cluster 2
        ])
        # A point cloud forming a circle
        self.points2 = np.array([
            [np.cos(t), np.sin(t)] for t in np.linspace(0, 2 * np.pi, 10, endpoint=False)
        ])

    def test_construct_vietoris_rips(self):
        rips_complex = construct_vietoris_rips(self.points2, max_edge_length=1.0, max_dim=2)
        self.assertIsInstance(rips_complex, list)
        # Check for at least vertices and edges
        self.assertGreater(len(rips_complex), len(self.points2))
        
        # Test with invalid input
        with self.assertRaises(TypeError):
            construct_vietoris_rips("not an array", 1.0)
        with self.assertRaises(ValueError):
            construct_vietoris_rips(self.points2, -1.0)

    def test_compute_persistent_homology(self):
        result = compute_persistent_homology(self.points1, max_dim=1)
        self.assertIsInstance(result, dict)
        self.assertIn('dgms', result)
        self.assertIsInstance(result['dgms'], list)
        self.assertEqual(len(result['dgms']), 2) # H0 and H1

        # Test with distance matrix
        dist_matrix = np.linalg.norm(self.points1[:, np.newaxis, :] - self.points1[np.newaxis, :, :], axis=-1)
        result_dist = compute_persistent_homology(dist_matrix, max_dim=1)
        self.assertEqual(len(result_dist['dgms']), 2)

    def test_calculate_tda_metrics(self):
        # Test with two components
        result1 = compute_persistent_homology(self.points1, max_dim=1)
        metrics1 = calculate_tda_metrics(result1['dgms'])
        self.assertEqual(metrics1['component_count'], 2)
        
        # Test with one component and one loop
        result2 = compute_persistent_homology(self.points2, max_dim=1)
        metrics2 = calculate_tda_metrics(result2['dgms'])
        self.assertEqual(metrics2['component_count'], 1)
        self.assertGreater(metrics2['total_b1_persistence'], 0)

        # Test with invalid input
        with self.assertRaises(TypeError):
            calculate_tda_metrics("not a list")

if __name__ == '__main__':
    unittest.main()
