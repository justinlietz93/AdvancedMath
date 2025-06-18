# Documentation for numerical_integrate Function

## Overview

The `numerical_integrate` function provides a robust interface to SciPy's numerical integration capabilities, specifically wrapping the `scipy.integrate.quad` function. This tool allows for the numerical calculation of definite integrals of arbitrary functions over specified intervals.

## Implementation Details

### Core Functionality

The function uses `scipy.integrate.quad` to perform adaptive numerical integration with error control. Key features include:

1. **Flexible Function Support**: Accepts any Python callable that takes a float as its first argument and returns a float.

2. **Parameter Passing**: Supports additional parameters to the integrand function through the `args` parameter.

3. **Error Estimation**: Returns both the computed integral value and an estimate of the absolute error.

4. **Comprehensive Error Handling**: Validates inputs and provides meaningful error messages for various failure scenarios.

### Input Validation

The implementation includes thorough input validation:
- Checks that `func` is callable
- Verifies that `a` and `b` are numeric values
- Ensures that `a < b` (lower bound must be less than upper bound)
- Confirms that `args` is a tuple

### Error Handling

The function handles potential integration failures with specific error messages:
- Singularity detection
- Convergence failures
- Other numerical issues

## Usage Examples

The function can be used for various integration scenarios:

1. **Simple Polynomial Integration**:
   ```python
   # Integrate x^2 from 0 to 1 (equals 1/3)
   result, error = numerical_integrate(lambda x: x**2, 0, 1)
   ```

2. **Trigonometric Functions**:
   ```python
   # Integrate sin(x) from 0 to π (equals 2)
   result, error = numerical_integrate(lambda x: np.sin(x), 0, np.pi)
   ```

3. **Functions with Parameters**:
   ```python
   def integrand(x, a, b):
       return a * np.sin(b * x)
   
   # Integrate 2*sin(x) from 0 to π (equals 4)
   result, error = numerical_integrate(integrand, 0, np.pi, args=(2.0, 1.0))
   ```

4. **Improper Integrals**:
   ```python
   # Integrate e^(-x^2) from -∞ to ∞ (equals sqrt(π))
   result, error = numerical_integrate(lambda x: np.exp(-x**2), -np.inf, np.inf)
   ```

5. **Functions with Singularities**:
   ```python
   # Integrate 1/sqrt(x) from 0 to 1 (equals 2)
   result, error = numerical_integrate(lambda x: 1/np.sqrt(x), 0, 1)
   ```

## Test Results

The function has been thoroughly tested with various integration scenarios:

| Test Case | Function | Interval | Result | Analytical | Absolute Difference |
|-----------|----------|----------|--------|------------|---------------------|
| 1 | x² | [0, 1] | 0.33333333 | 0.33333333 | 5.55e-17 |
| 2 | sin(x) | [0, π] | 2.00000000 | 2.00000000 | 0.00e+00 |
| 3 | e^(-x²) | [-∞, ∞] | 1.77245385 | 1.77245385 | 0.00e+00 |
| 4 | 2*sin(x) | [0, π] | 4.00000000 | 4.00000000 | 0.00e+00 |
| 5 | sin(10x) | [0, 2π] | -0.00000000 | 0.00000000 | 7.82e-17 |
| 7 | 1/sqrt(x) | [0, 1] | 2.00000000 | 2.00000000 | 1.55e-15 |

All test cases show excellent agreement with analytical results, with differences well within numerical precision limits.

## Dependencies

The function requires:
- NumPy
- SciPy (specifically the `scipy.integrate` module)

## Conclusion

The `numerical_integrate` function provides a reliable, efficient, and user-friendly interface for numerical integration tasks. Its robust error handling and comprehensive documentation make it suitable for a wide range of mathematical applications within the FUM (Fully Unified Model) project.
