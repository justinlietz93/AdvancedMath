"""
Unit tests for the symbolic_differentiation module.

This script contains tests to verify the functionality of the symbolic_differentiate function.
"""

import unittest
import sympy
from sympy import symbols, sin, cos, exp, log
from src import symbolic_differentiate

class TestSymbolicDifferentiation(unittest.TestCase):
    """Test cases for the symbolic_differentiate function."""
    
    def test_polynomial_string(self):
        """Test differentiation of polynomial expressions provided as strings."""
        # First derivative of x^2 + 2x + 1 is 2x + 2
        result = symbolic_differentiate("x**2 + 2*x + 1", "x")
        self.assertEqual(str(result), "2*x + 2")
        
        # First derivative of x^3 is 3x^2
        result = symbolic_differentiate("x**3", "x")
        self.assertEqual(str(result), "3*x**2")
    
    def test_polynomial_sympy(self):
        """Test differentiation of polynomial expressions provided as SymPy objects."""
        x = symbols('x')
        expr = x**2 + 2*x + 1
        
        result = symbolic_differentiate(expr, x)
        self.assertEqual(result, 2*x + 2)
    
    def test_trigonometric(self):
        """Test differentiation of trigonometric functions."""
        # d/dx(sin(x)) = cos(x)
        result = symbolic_differentiate("sin(x)", "x")
        self.assertEqual(str(result), "cos(x)")
        
        # d/dx(cos(x)) = -sin(x)
        result = symbolic_differentiate("cos(x)", "x")
        self.assertEqual(str(result), "-sin(x)")
    
    def test_exponential(self):
        """Test differentiation of exponential functions."""
        # d/dx(e^x) = e^x
        result = symbolic_differentiate("exp(x)", "x")
        self.assertEqual(str(result), "exp(x)")
        
        # d/dx(a^x) = a^x * ln(a)
        result = symbolic_differentiate("2**x", "x")
        self.assertEqual(str(result), "2**x*log(2)")
    
    def test_logarithmic(self):
        """Test differentiation of logarithmic functions."""
        # d/dx(ln(x)) = 1/x
        result = symbolic_differentiate("log(x)", "x")
        self.assertEqual(str(result), "1/x")
    
    def test_higher_order(self):
        """Test higher-order derivatives."""
        # d^2/dx^2(x^3) = 6x
        result = symbolic_differentiate("x**3", "x", order=2)
        self.assertEqual(str(result), "6*x")
        
        # d^3/dx^3(x^3) = 6
        result = symbolic_differentiate("x**3", "x", order=3)
        self.assertEqual(str(result), "6")
        
        # d^4/dx^4(x^3) = 0
        result = symbolic_differentiate("x**3", "x", order=4)
        self.assertEqual(str(result), "0")
    
    def test_partial_derivatives(self):
        """Test partial derivatives with multiple variables."""
        # ∂/∂x(x^2 * y^3) = 2x * y^3
        result = symbolic_differentiate("x**2 * y**3", "x")
        self.assertEqual(str(result), "2*x*y**3")
        
        # ∂/∂y(x^2 * y^3) = x^2 * 3y^2
        result = symbolic_differentiate("x**2 * y**3", "y")
        self.assertEqual(str(result), "3*x**2*y**2")
        
        # ∂^2/∂x∂y(x^2 * y^3) = 2x * 3y^2 = 6xy^2
        result = symbolic_differentiate("x**2 * y**3", ["x", "y"])
        self.assertEqual(str(result), "6*x*y**2")
    
    def test_variable_not_in_expression(self):
        """Test differentiation with respect to a variable not in the expression."""
        # d/dz(x^2 + y^2) = 0
        result = symbolic_differentiate("x**2 + y**2", "z")
        self.assertEqual(str(result), "0")
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Invalid expression syntax
        with self.assertRaises(sympy.SympifyError):
            symbolic_differentiate("x^2 + 1", "x")  # ^ is not valid in SymPy
        
        # Invalid order
        with self.assertRaises(ValueError):
            symbolic_differentiate("x**2", "x", order=0)
        
        with self.assertRaises(ValueError):
            symbolic_differentiate("x**2", "x", order=-1)
        
        # Empty variable list
        with self.assertRaises(ValueError):
            symbolic_differentiate("x**2", [])
        
        # Multiple variables with one not in expression
        with self.assertRaises(ValueError):
            symbolic_differentiate("x**2", ["x", "z"])
        
        # Invalid variable type
        with self.assertRaises(TypeError):
            symbolic_differentiate("x**2", 5)

if __name__ == "__main__":
    unittest.main(verbosity=2)
