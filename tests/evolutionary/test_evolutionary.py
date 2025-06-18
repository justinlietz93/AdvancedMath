import unittest
import numpy as np
from src.evolutionary import apply_mutation, apply_recombination

class TestEvolutionary(unittest.TestCase):

    def setUp(self):
        self.weights1 = np.ones((10, 10))
        self.weights2 = np.zeros((10, 10))

    def test_apply_mutation(self):
        mutated_weights = apply_mutation(self.weights1, mutation_rate=1.0, mutation_scale=0.1)
        self.assertFalse(np.allclose(self.weights1, mutated_weights))
        self.assertEqual(self.weights1.shape, mutated_weights.shape)

    def test_apply_recombination(self):
        recombined_weights = apply_recombination(self.weights1, self.weights2, recombination_prob=0.5)
        self.assertTrue(np.any(recombined_weights == 1))
        self.assertTrue(np.any(recombined_weights == 0))
        self.assertEqual(self.weights1.shape, recombined_weights.shape)

if __name__ == '__main__':
    unittest.main()
