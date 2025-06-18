import unittest
import numpy as np
from src.dynamical_systems import find_fixed_points, calculate_jacobian, analyze_stability

class TestDynamicalSystems(unittest.TestCase):

    def setUp(self):
        # A simple linear system: dx/dt = -x, dy/dt = -2y
        self.linear_system = lambda y: np.array([-y[0], -2*y[1]])
        self.fixed_point = np.array([0, 0])

    def test_find_fixed_points(self):
        found_points = find_fixed_points(self.linear_system, [np.array([1, 1])])
        self.assertTrue(np.allclose(found_points[0], self.fixed_point))

    def test_calculate_jacobian(self):
        jacobian = calculate_jacobian(self.linear_system, self.fixed_point)
        expected_jacobian = np.array([[-1, 0], [0, -2]])
        self.assertTrue(np.allclose(jacobian, expected_jacobian))

    def test_analyze_stability(self):
        jacobian = calculate_jacobian(self.linear_system, self.fixed_point)
        analysis = analyze_stability(jacobian)
        self.assertEqual(analysis['stability_type'], "Stable Node")
        self.assertTrue(np.all(analysis['eigenvalues'] < 0))

if __name__ == '__main__':
    unittest.main()
