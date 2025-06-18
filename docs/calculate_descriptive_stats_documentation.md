# Descriptive Statistics Calculator Documentation

## Overview

The `calculate_descriptive_stats` function provides a comprehensive solution for computing descriptive statistics on numerical datasets. It leverages SciPy's efficient `stats.describe` function to calculate multiple statistics in a single pass through the data, making it both performant and convenient for data analysis tasks.

This documentation provides detailed information about the function, its capabilities, usage scenarios, and limitations.

## Purpose

Descriptive statistics summarize and quantify the main features of a dataset, providing insights into its central tendency, dispersion, and shape. These statistics are fundamental in data analysis, helping to understand the underlying distribution of data before applying more complex analytical techniques.

The `calculate_descriptive_stats` function is designed to be a one-stop solution for calculating the most commonly used descriptive statistics, returning them in a standardized dictionary format that's easy to work with.

## Function Signature and Parameters

```python
def calculate_descriptive_stats(data, nan_policy='propagate'):
    """
    Calculate descriptive statistics for a given dataset.
    """
```

### Parameters

#### `data` (array_like)
- **Description**: The input dataset for which statistics will be calculated.
- **Type**: Any array-like object (list, tuple, NumPy array, etc.) that can be converted to a 1D NumPy array of numbers.
- **Requirements**: Must be convertible to a 1D numeric array and cannot be empty.
- **Examples**: `[1, 2, 3, 4, 5]`, `np.array([1.5, 2.5, 3.5])`, `(10, 20, 30, 40)`

#### `nan_policy` (str, optional)
- **Description**: Defines how to handle NaN (Not a Number) values in the input data.
- **Default**: `'propagate'`
- **Valid values**:
  - `'propagate'`: Returns NaN for statistics when NaN values are present in the data
  - `'omit'`: Ignores NaN values when computing statistics
  - `'raise'`: Raises an error if NaN values are present in the data
- **Usage**: Choose based on how you want to handle missing or invalid data points.

## Return Value

The function returns a dictionary containing the following descriptive statistics:

| Key | Description | Formula/Method |
|-----|-------------|----------------|
| `'count'` | Number of observations | Count of non-NaN values (if `nan_policy='omit'`) |
| `'mean'` | Arithmetic mean | Sum of values divided by count |
| `'stdev'` | Standard deviation (sample) | Square root of variance |
| `'variance'` | Variance (sample) | Average of squared deviations from the mean |
| `'min'` | Minimum value | Smallest value in the dataset |
| `'max'` | Maximum value | Largest value in the dataset |
| `'skewness'` | Skewness | Third standardized moment, measures asymmetry |
| `'kurtosis'` | Kurtosis | Fourth standardized moment, measures "tailedness" (Fisher's definition: normal = 0.0) |

## Statistical Measures Explained

### Central Tendency
- **Mean**: The average value, sensitive to outliers.
- **Interpretation**: Represents the "center" of the data.

### Dispersion
- **Variance**: Measures how spread out the data is from the mean.
- **Standard Deviation**: Square root of variance, in the same units as the original data.
- **Min/Max**: The range of values in the dataset.
- **Interpretation**: Larger values indicate more variability in the data.

### Shape
- **Skewness**: Measures the asymmetry of the probability distribution.
  - Positive skewness: Right tail is longer (values extend more to the right)
  - Negative skewness: Left tail is longer (values extend more to the left)
  - Zero skewness: Symmetric distribution
  - **Interpretation**: Indicates which side of the distribution has more extreme values.

- **Kurtosis**: Measures the "tailedness" of the probability distribution.
  - Positive kurtosis: Heavy-tailed distribution (more outliers)
  - Negative kurtosis: Light-tailed distribution (fewer outliers)
  - Zero kurtosis: Normal distribution (Fisher's definition)
  - **Interpretation**: Indicates the presence of outliers or extreme values.

## Error Handling

The function includes comprehensive error handling for various edge cases:

### Input Validation
- **None data**: Raises `ValueError` if input data is None.
- **Non-numeric data**: Raises `TypeError` if input cannot be converted to a numeric array.
- **Multi-dimensional data**: Raises `ValueError` if input is not 1-dimensional.
- **Empty data**: Raises `ValueError` if input array is empty.
- **Invalid nan_policy**: Raises `ValueError` if nan_policy is not one of the valid options.

### Special Cases
- **NaN values**: Handled according to the specified `nan_policy`.
- **Zero variance**: Special handling for constant data (all values identical), setting variance and standard deviation to 0, and skewness and kurtosis to 0.

## Implementation Details

The function uses `scipy.stats.describe` internally, which efficiently calculates multiple statistics in a single pass through the data. This approach is more performant than calculating each statistic separately, especially for large datasets.

The implementation follows these steps:
1. Validate input parameters
2. Convert input to a NumPy array
3. Perform additional validation checks
4. Call `scipy.stats.describe` to calculate statistics
5. Process the result into a standardized dictionary
6. Handle special cases and errors

## Limitations

1. **Numerical Precision**:
   - Results are subject to floating-point precision limitations.
   - Very large or small values may lead to numerical instability.

2. **Statistical Assumptions**:
   - Skewness and kurtosis calculations assume sufficient data points (generally n > 8).
   - For small samples, these higher moments may not be reliable.

3. **Performance**:
   - While efficient for most datasets, very large arrays (millions of elements) may require significant memory.

4. **NaN Handling**:
   - With `nan_policy='propagate'`, a single NaN can cause many statistics to be NaN.
   - With `nan_policy='omit'`, statistics are calculated only on non-NaN values, which may not represent the complete dataset.

## Related NumPy/SciPy Functions

- `scipy.stats.describe`: The core function used internally
- `numpy.mean`, `numpy.std`, `numpy.var`: Individual statistics functions
- `scipy.stats.skew`, `scipy.stats.kurtosis`: Individual shape statistics
- `pandas.DataFrame.describe`: Similar functionality in pandas for DataFrames

## Dependencies

- **NumPy**: For array operations and basic statistics
- **SciPy**: For the `stats.describe` function that efficiently calculates multiple statistics
