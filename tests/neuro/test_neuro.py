import unittest
import numpy as np
from src.neuro import (
    apply_stdp,
    apply_stc,
    calculate_stabilized_reward,
    apply_quadratic_stdp_modulation
)

class TestNeuro(unittest.TestCase):

    def test_apply_stdp(self):
        # This is already tested in test_apply_stdp.py, but we can add a simple check
        new_weight, _ = apply_stdp([10], [15], 0.5)
        self.assertNotEqual(new_weight, 0.5)

    def test_apply_stc(self):
        new_weight, _, _ = apply_stc(0.5, 0.1, 0.2, 0.8)
        self.assertNotEqual(new_weight, 0.5)

    def test_calculate_stabilized_reward(self):
        reward = calculate_stabilized_reward(0.1, 0.2, 0.3, 0.4, 0.5)
        self.assertIsInstance(reward, float)

    def test_apply_quadratic_stdp_modulation(self):
        modulation = apply_quadratic_stdp_modulation(total_reward=0.5)
        self.assertIsInstance(modulation, float)

if __name__ == '__main__':
    unittest.main()
