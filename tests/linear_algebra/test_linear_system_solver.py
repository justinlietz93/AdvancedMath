import unittest
import numpy as np
from src import linear_system_solver

class TestLinearSystemSolver(unittest.TestCase):
    """Test cases for the linear_system_solver function."""
    
    def test_simple_2x2(self):
        """Test a simple 2x2 system with known solution."""
        A = np.array([[3, 2], [1, 1]])
        b = np.array([7, 3])
        expected = np.array([1.0, 2.0])
        
        result = linear_system_solver(A, b)
        
        self.assertTrue(np.allclose(result, expected))
        self.assertTrue(np.allclose(np.dot(A, result), b))
    
    def test_3x3_system(self):
        """Test a 3x3 system with known solution."""
        A = np.array([[2, 1, 1], [3, 5, 2], [1, 3, 5]])
        b = np.array([5, 15, 14])
        
        result = linear_system_solver(A, b)
        
        # Verify that the result satisfies the original equation Ax = b
        self.assertTrue(np.allclose(np.dot(A, result), b))
    
    def test_list_inputs(self):
        """Test with list inputs instead of NumPy arrays."""
        A = [[4, 2], [2, 5]]
        b = [10, 16]
        
        result = linear_system_solver(A, b)
        
        # Verify that the result satisfies the original equation Ax = b
        self.assertTrue(np.allclose(np.dot(A, result), b))
    
    def test_integer_solution(self):
        """Test a system with integer solution."""
        A = np.array([[5, 3], [2, 1]])
        b = np.array([11, 5])
        
        result = linear_system_solver(A, b)
        
        # Verify that the result satisfies the original equation Ax = b
        self.assertTrue(np.allclose(np.dot(A, result), b))
    
    def test_float_solution(self):
        """Test a system with floating-point solution."""
        A = np.array([[2.5, 1.5], [1.0, 1.0]])
        b = np.array([7.0, 3.0])
        
        result = linear_system_solver(A, b)
        
        # Verify that the result satisfies the original equation Ax = b
        self.assertTrue(np.allclose(np.dot(A, result), b))
    
    def test_large_system(self):
        """Test a larger system (10x10)."""
        n = 10
        # Create a random non-singular matrix
        np.random.seed(42)  # For reproducibility
        A = np.random.rand(n, n)
        # Ensure it's not singular by adding identity
        A = A + np.eye(n)
        
        # Create a random solution vector
        x_true = np.random.rand(n)
        
        # Compute right-hand side
        b = np.dot(A, x_true)
        
        # Solve the system
        result = linear_system_solver(A, b)
        
        # Check if the solution is correct
        self.assertTrue(np.allclose(result, x_true))
        self.assertTrue(np.allclose(np.dot(A, result), b))
    
    def test_error_non_square_matrix(self):
        """Test error handling for non-square matrix."""
        A = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3 matrix
        b = np.array([7, 8])
        
        with self.assertRaises(ValueError) as context:
            linear_system_solver(A, b)
        
        self.assertTrue("must be square" in str(context.exception))
    
    def test_error_incompatible_dimensions(self):
        """Test error handling for incompatible dimensions."""
        A = np.array([[1, 2], [3, 4]])  # 2x2 matrix
        b = np.array([5, 6, 7])  # 3-element vector
        
        with self.assertRaises(ValueError) as context:
            linear_system_solver(A, b)
        
        self.assertTrue("Incompatible shapes" in str(context.exception))
    
    def test_error_singular_matrix(self):
        """Test error handling for singular matrix."""
        A = np.array([[1, 2], [2, 4]])  # Singular matrix (row 2 = 2*row 1)
        b = np.array([3, 6])
        
        with self.assertRaises(np.linalg.LinAlgError) as context:
            linear_system_solver(A, b)
        
        self.assertTrue("singular" in str(context.exception).lower())
    
    def test_error_non_numeric_data(self):
        """Test error handling for non-numeric data."""
        A = np.array([['a', 'b'], ['c', 'd']])
        b = np.array(['e', 'f'])
        
        with self.assertRaises(TypeError) as context:
            linear_system_solver(A, b)
        
        self.assertTrue("numeric" in str(context.exception).lower())
    
    def test_error_none_inputs(self):
        """Test error handling for None inputs."""
        with self.assertRaises(ValueError) as context:
            linear_system_solver(None, np.array([1, 2]))
        
        self.assertTrue("cannot be None" in str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            linear_system_solver(np.array([[1, 2], [3, 4]]), None)
        
        self.assertTrue("cannot be None" in str(context.exception))
    
    def test_error_empty_arrays(self):
        """Test error handling for empty arrays."""
        A = np.zeros((0, 0))
        b = np.array([])
        
        with self.assertRaises(ValueError) as context:
            linear_system_solver(A, b)
        
        self.assertTrue("Empty arrays" in str(context.exception))

    def test_error_nan_infinity(self):
        """Test error handling for NaN or infinity values."""
        A = np.array([[1, 2], [3, np.nan]])
        b = np.array([4, 5])
        
        with self.assertRaises(ValueError) as context:
            linear_system_solver(A, b)
        
        self.assertTrue("NaN or infinity" in str(context.exception))
        
        A = np.array([[1, 2], [3, 4]])
        b = np.array([np.inf, 5])
        
        with self.assertRaises(ValueError) as context:
            linear_system_solver(A, b)
        
        self.assertTrue("NaN or infinity" in str(context.exception))

if __name__ == '__main__':
    unittest.main()
