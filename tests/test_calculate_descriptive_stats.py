import unittest
import numpy as np
from calculate_descriptive_stats import calculate_descriptive_stats

class TestCalculateDescriptiveStats(unittest.TestCase):
    """Test cases for the calculate_descriptive_stats function."""
    
    def test_basic_list(self):
        """Test with a simple list of integers."""
        data = [1, 2, 3, 4, 5]
        result = calculate_descriptive_stats(data)
        
        self.assertEqual(result['count'], 5)
        self.assertEqual(result['mean'], 3.0)
        self.assertEqual(result['variance'], 2.5)
        self.assertEqual(result['stdev'], np.sqrt(2.5))
        self.assertEqual(result['min'], 1)
        self.assertEqual(result['max'], 5)
        self.assertAlmostEqual(result['skewness'], 0.0, places=10)
        self.assertAlmostEqual(result['kurtosis'], -1.3, places=1)
    
    def test_numpy_array(self):
        """Test with a NumPy array."""
        data = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        result = calculate_descriptive_stats(data)
        
        self.assertEqual(result['count'], 5)
        self.assertEqual(result['mean'], 3.5)
        self.assertEqual(result['variance'], 2.5)
        self.assertEqual(result['stdev'], np.sqrt(2.5))
        self.assertEqual(result['min'], 1.5)
        self.assertEqual(result['max'], 5.5)
    
    def test_tuple_input(self):
        """Test with a tuple input."""
        data = (10, 20, 30, 40, 50)
        result = calculate_descriptive_stats(data)
        
        self.assertEqual(result['count'], 5)
        self.assertEqual(result['mean'], 30.0)
        self.assertEqual(result['min'], 10)
        self.assertEqual(result['max'], 50)
    
    def test_nan_policy_propagate(self):
        """Test with NaN values and nan_policy='propagate'."""
        data = [1.0, 2.0, np.nan, 4.0, 5.0]
        result = calculate_descriptive_stats(data, nan_policy='propagate')
        
        self.assertEqual(result['count'], 4)  # NaN is not counted
        self.assertTrue(np.isnan(result['mean']))
        self.assertTrue(np.isnan(result['variance']))
        self.assertTrue(np.isnan(result['stdev']))
        self.assertEqual(result['min'], 1.0)
        self.assertEqual(result['max'], 5.0)
    
    def test_nan_policy_omit(self):
        """Test with NaN values and nan_policy='omit'."""
        data = [1.0, 2.0, np.nan, 4.0, 5.0]
        result = calculate_descriptive_stats(data, nan_policy='omit')
        
        self.assertEqual(result['count'], 4)  # NaN is not counted
        self.assertEqual(result['mean'], 3.0)  # (1+2+4+5)/4 = 3.0
        self.assertEqual(result['min'], 1.0)
        self.assertEqual(result['max'], 5.0)
    
    def test_nan_policy_raise(self):
        """Test with NaN values and nan_policy='raise'."""
        data = [1.0, 2.0, np.nan, 4.0, 5.0]
        
        with self.assertRaises(ValueError):
            calculate_descriptive_stats(data, nan_policy='raise')
    
    def test_zero_variance(self):
        """Test with constant data (zero variance)."""
        data = [5, 5, 5, 5, 5]
        result = calculate_descriptive_stats(data)
        
        self.assertEqual(result['count'], 5)
        self.assertEqual(result['mean'], 5.0)
        self.assertEqual(result['variance'], 0.0)
        self.assertEqual(result['stdev'], 0.0)
        self.assertEqual(result['min'], 5)
        self.assertEqual(result['max'], 5)
        self.assertEqual(result['skewness'], 0.0)
        self.assertEqual(result['kurtosis'], 0.0)
    
    def test_single_value(self):
        """Test with a single value."""
        data = [42]
        result = calculate_descriptive_stats(data)
        
        self.assertEqual(result['count'], 1)
        self.assertEqual(result['mean'], 42.0)
        self.assertEqual(result['variance'], 0.0)
        self.assertEqual(result['stdev'], 0.0)
        self.assertEqual(result['min'], 42)
        self.assertEqual(result['max'], 42)
        self.assertEqual(result['skewness'], 0.0)
        self.assertEqual(result['kurtosis'], 0.0)
    
    def test_error_none_data(self):
        """Test error handling for None data."""
        with self.assertRaises(ValueError):
            calculate_descriptive_stats(None)
    
    def test_error_empty_data(self):
        """Test error handling for empty data."""
        with self.assertRaises(ValueError):
            calculate_descriptive_stats([])
    
    def test_error_non_numeric_data(self):
        """Test error handling for non-numeric data."""
        with self.assertRaises(TypeError):
            calculate_descriptive_stats(['a', 'b', 'c'])
    
    def test_error_multidimensional_data(self):
        """Test error handling for multi-dimensional data."""
        with self.assertRaises(ValueError):
            calculate_descriptive_stats([[1, 2], [3, 4]])
    
    def test_error_invalid_nan_policy(self):
        """Test error handling for invalid nan_policy."""
        with self.assertRaises(ValueError):
            calculate_descriptive_stats([1, 2, 3], nan_policy='invalid')

if __name__ == '__main__':
    unittest.main()
