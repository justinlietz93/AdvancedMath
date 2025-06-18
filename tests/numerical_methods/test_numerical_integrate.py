import unittest
import numpy as np
import matplotlib.pyplot as plt
from src import numerical_integrate

class TestNumericalIntegrate(unittest.TestCase):
    """Test cases for the numerical_integrate function."""

    def test_polynomial(self):
        """Test a simple polynomial (x^2 from 0 to 1, analytical result = 1/3)."""
        result, error = numerical_integrate(lambda x: x**2, 0, 1)
        analytical = 1/3
        self.assertAlmostEqual(result, analytical, places=7)

    def test_trigonometric(self):
        """Test a trigonometric function (sin(x) from 0 to pi, analytical result = 2)."""
        result, error = numerical_integrate(lambda x: np.sin(x), 0, np.pi)
        analytical = 2.0
        self.assertAlmostEqual(result, analytical, places=7)

    def test_exponential(self):
        """Test an exponential function (e^(-x^2) from -inf to inf, analytical result = sqrt(pi))."""
        result, error = numerical_integrate(lambda x: np.exp(-x**2), -np.inf, np.inf)
        analytical = np.sqrt(np.pi)
        self.assertAlmostEqual(result, analytical, places=7)

    def test_with_parameters(self):
        """Test a function with additional parameters."""
        def integrand(x, a, b):
            return a * np.sin(b * x)
        result, error = numerical_integrate(integrand, 0, np.pi, args=(2.0, 1.0))
        analytical = 4.0
        self.assertAlmostEqual(result, analytical, places=7)

    def test_oscillatory(self):
        """Test an oscillatory function (sin(10x) from 0 to 2pi)."""
        result, error = numerical_integrate(lambda x: np.sin(10*x), 0, 2*np.pi)
        analytical = 0.0
        self.assertAlmostEqual(result, analytical, places=7)

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        with self.assertRaises(TypeError):
            numerical_integrate("not_a_function", 0, 1)
        with self.assertRaises(ValueError):
            numerical_integrate(lambda x: x**2, 1, 0)

    def test_singularity(self):
        """Test a function with a singularity (handled by quad)."""
        result, error = numerical_integrate(lambda x: 1/np.sqrt(x), 0, 1)
        analytical = 2.0
        self.assertAlmostEqual(result, analytical, places=7)

if __name__ == '__main__':
    unittest.main()
