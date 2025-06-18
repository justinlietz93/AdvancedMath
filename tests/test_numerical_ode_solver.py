"""
Unit tests for the numerical_ode_solver function.

This script contains tests to verify the functionality of the numerical_ode_solver function.
"""

import unittest
import numpy as np
from numerical_ode_solver import numerical_ode_solver

class TestNumericalOdeSolver(unittest.TestCase):
    """Test cases for the numerical_ode_solver function."""
    
    def test_exponential_decay(self):
        """Test solving a simple exponential decay ODE."""
        def exponential_decay(t, y, rate_constant):
            return -rate_constant * y
        
        t_span = (0, 10)
        y0 = [1.0]
        rate_constant = 0.1
        
        sol = numerical_ode_solver(exponential_decay, t_span, y0, args=(rate_constant,))
        
        # Check if the solution is close to the analytical solution
        t_final = sol.t[-1]
        y_numerical = sol.y[0, -1]
        y_analytical = y0[0] * np.exp(-rate_constant * t_final)
        
        self.assertTrue(sol.success)
        self.assertAlmostEqual(y_numerical, y_analytical, places=3)
    
    def test_harmonic_oscillator(self):
        """Test solving a harmonic oscillator ODE."""
        def harmonic_oscillator(t, y, omega):
            x, v = y
            dxdt = v
            dvdt = -(omega**2) * x
            return [dxdt, dvdt]
        
        omega = 2.0
        t_span = (0, 10)
        y0 = [1.0, 0.0]  # Initial position and velocity
        
        sol = numerical_ode_solver(harmonic_oscillator, t_span, y0, args=(omega,))
        
        # Check if the solution is close to the analytical solution at several points
        for i in range(min(5, len(sol.t))):
            t = sol.t[i]
            x_numerical = sol.y[0, i]
            v_numerical = sol.y[1, i]
            
            x_analytical = y0[0] * np.cos(omega * t)
            v_analytical = -y0[0] * omega * np.sin(omega * t)
            
            self.assertAlmostEqual(x_numerical, x_analytical, places=2)
            self.assertAlmostEqual(v_numerical, v_analytical, places=2)
    
    def test_lotka_volterra(self):
        """Test solving the Lotka-Volterra predator-prey model."""
        def lotka_volterra(t, z, a, b, c, d):
            x, y = z
            dx_dt = a * x - b * x * y
            dy_dt = -c * y + d * x * y
            return [dx_dt, dy_dt]
        
        t_span = (0, 1)  # Short time span for testing
        y0 = [10, 5]
        params = (1.5, 1, 3, 1)
        
        sol = numerical_ode_solver(lotka_volterra, t_span, y0, args=params)
        
        # Check if the solution has the expected shape and properties
        self.assertTrue(sol.success)
        self.assertEqual(sol.y.shape[0], 2)  # Two variables
        self.assertTrue(len(sol.t) > 0)  # Some time points were computed
        
        # Check conservation properties (not exact but should be close)
        # For certain parameter values, there are conservation laws
        if params == (1, 1, 1, 1):
            for i in range(len(sol.t)):
                x, y = sol.y[0, i], sol.y[1, i]
                # The quantity x*y*e^(-(x+y)) is approximately conserved
                conserved = x * y * np.exp(-(x + y))
                initial_conserved = y0[0] * y0[1] * np.exp(-(y0[0] + y0[1]))
                self.assertAlmostEqual(conserved, initial_conserved, places=1)
    
    def test_stiff_system(self):
        """Test solving a stiff ODE system with different methods."""
        def stiff_system(t, y, lambda1, lambda2):
            return [lambda1 * y[0], lambda2 * y[1]]
        
        lambda1 = -0.1
        lambda2 = -100.0  # Not too stiff for testing
        t_span = (0, 10)
        y0 = [1.0, 1.0]
        
        # Test with different methods
        methods = ['RK45', 'BDF', 'LSODA']
        
        for method in methods:
            sol = numerical_ode_solver(
                stiff_system, t_span, y0, 
                args=(lambda1, lambda2), 
                method=method
            )
            
            # Check if the solution is close to the analytical solution
            t_final = sol.t[-1]
            y1_numerical = sol.y[0, -1]
            y2_numerical = sol.y[1, -1]
            
            y1_analytical = np.exp(lambda1 * t_final)
            y2_analytical = np.exp(lambda2 * t_final)
            
            self.assertTrue(sol.success)
            self.assertAlmostEqual(y1_numerical, y1_analytical, places=2)
            # For the stiff component, we use a lower precision
            self.assertAlmostEqual(y2_numerical, y2_analytical, places=1)
    
    def test_input_validation(self):
        """Test input validation for various error conditions."""
        def simple_ode(t, y):
            return -y
        
        # Test invalid t_span
        with self.assertRaises(ValueError):
            numerical_ode_solver(simple_ode, (10, 0), [1.0])  # End time before start time
        
        # Test invalid y0 type
        with self.assertRaises(TypeError):
            numerical_ode_solver(simple_ode, (0, 10), "not_an_array")
        
        # Test invalid method
        with self.assertRaises(ValueError):
            numerical_ode_solver(simple_ode, (0, 10), [1.0], method="INVALID_METHOD")
        
        # Test invalid tolerance
        with self.assertRaises(ValueError):
            numerical_ode_solver(simple_ode, (0, 10), [1.0], rtol=-1.0)
        
        # Test ODE function with wrong return shape
        def wrong_shape_ode(t, y):
            return [-y[0], -y[0]]  # Returns 2 values for a 1D input
        
        with self.assertRaises(ValueError):
            numerical_ode_solver(wrong_shape_ode, (0, 10), [1.0])
    
    def test_t_eval_parameter(self):
        """Test the t_eval parameter for specifying output times."""
        def exponential_decay(t, y):
            return -0.1 * y
        
        t_span = (0, 10)
        y0 = [1.0]
        t_eval = np.linspace(0, 10, 11)  # 11 evenly spaced points
        
        sol = numerical_ode_solver(exponential_decay, t_span, y0, t_eval=t_eval)
        
        # Check if the output times match t_eval
        self.assertEqual(len(sol.t), len(t_eval))
        for i in range(len(t_eval)):
            self.assertAlmostEqual(sol.t[i], t_eval[i])

if __name__ == "__main__":
    unittest.main(verbosity=2)
