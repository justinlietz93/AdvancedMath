import unittest
import numpy as np
from src.causal_inference import granger_causality, calculate_transfer_entropy

class TestCausalInference(unittest.TestCase):

    def setUp(self):
        # Create two time series where x "causes" y
        self.x = np.random.rand(100)
        self.y = np.roll(self.x, 1) + np.random.rand(100) * 0.1
        self.data = np.vstack([self.x, self.y]).T

    def test_granger_causality(self):
        # This is a statistical test, so we can't assert a specific outcome,
        # but we can check that it runs without error and returns the correct format.
        result = granger_causality(self.data, max_lag=2)
        self.assertIsInstance(result, dict)
        self.assertIn(1, result)
        self.assertIn('ssr_chi2test', result[1][0])

    def test_transfer_entropy(self):
        # We expect the transfer entropy from x to y to be greater than from y to x
        te_xy = calculate_transfer_entropy(self.x, self.y)
        te_yx = calculate_transfer_entropy(self.y, self.x)
        self.assertGreater(te_xy, te_yx)

if __name__ == '__main__':
    unittest.main()
