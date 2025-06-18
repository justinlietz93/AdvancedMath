import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from calculate_descriptive_stats import calculate_descriptive_stats

def example_1_basic_usage():
    """
    Example 1: Basic usage with a simple list of numbers
    """
    print("\nExample 1: Basic Usage")
    print("---------------------")
    
    # Create a simple dataset
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # Calculate descriptive statistics
    stats = calculate_descriptive_stats(data)
    
    # Print the results
    print(f"Dataset: {data}")
    print("\nDescriptive Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

def example_2_numpy_array():
    """
    Example 2: Using a NumPy array with some outliers
    """
    print("\nExample 2: NumPy Array with Outliers")
    print("----------------------------------")
    
    # Create a dataset with outliers
    np.random.seed(42)  # For reproducibility
    data = np.random.normal(loc=50, scale=5, size=100)  # Normal distribution
    data = np.append(data, [80, 85, 20, 15])  # Add outliers
    
    # Calculate descriptive statistics
    stats = calculate_descriptive_stats(data)
    
    # Print selected statistics
    print(f"Dataset: Normal distribution (mean=50, std=5) with outliers")
    print(f"Number of observations: {stats['count']}")
    print(f"Mean: {stats['mean']:.2f}")
    print(f"Standard Deviation: {stats['stdev']:.2f}")
    print(f"Min: {stats['min']:.2f}")
    print(f"Max: {stats['max']:.2f}")
    print(f"Skewness: {stats['skewness']:.4f}")
    print(f"Kurtosis: {stats['kurtosis']:.4f}")
    
    # Visualize the data
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(stats['mean'], color='red', linestyle='dashed', linewidth=2, label=f"Mean: {stats['mean']:.2f}")
    plt.axvline(stats['mean'] + stats['stdev'], color='green', linestyle='dotted', linewidth=2, 
                label=f"Mean + StdDev: {stats['mean'] + stats['stdev']:.2f}")
    plt.axvline(stats['mean'] - stats['stdev'], color='green', linestyle='dotted', linewidth=2, 
                label=f"Mean - StdDev: {stats['mean'] - stats['stdev']:.2f}")
    plt.title('Distribution with Outliers')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('distribution_with_outliers.png')
    print("Saved histogram to 'distribution_with_outliers.png'")

def example_3_handling_nan_values():
    """
    Example 3: Demonstrating different NaN handling policies
    """
    print("\nExample 3: Handling NaN Values")
    print("--------------------------")
    
    # Create a dataset with NaN values
    data = [1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0, 8.0]
    
    print(f"Dataset with NaNs: {data}")
    
    # Policy: 'propagate' (default)
    try:
        stats_propagate = calculate_descriptive_stats(data, nan_policy='propagate')
        print("\nWith nan_policy='propagate':")
        for key, value in stats_propagate.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"\nWith nan_policy='propagate': Error - {str(e)}")
    
    # Policy: 'omit'
    try:
        stats_omit = calculate_descriptive_stats(data, nan_policy='omit')
        print("\nWith nan_policy='omit':")
        for key, value in stats_omit.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"\nWith nan_policy='omit': Error - {str(e)}")
    
    # Policy: 'raise'
    try:
        stats_raise = calculate_descriptive_stats(data, nan_policy='raise')
        print("\nWith nan_policy='raise':")
        for key, value in stats_raise.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"\nWith nan_policy='raise': Error - {str(e)}")

def example_4_comparing_datasets():
    """
    Example 4: Comparing statistics between two datasets
    """
    print("\nExample 4: Comparing Datasets")
    print("-------------------------")
    
    # Create two datasets
    np.random.seed(42)  # For reproducibility
    data1 = np.random.normal(loc=50, scale=10, size=1000)  # Normal distribution
    data2 = np.random.exponential(scale=10, size=1000) + 30  # Exponential distribution
    
    # Calculate descriptive statistics
    stats1 = calculate_descriptive_stats(data1)
    stats2 = calculate_descriptive_stats(data2)
    
    # Print comparison
    print("Comparison of Normal vs. Exponential Distribution:")
    print(f"{'Statistic':<10} {'Normal':<15} {'Exponential':<15}")
    print("-" * 40)
    for key in stats1.keys():
        if isinstance(stats1[key], (int, float)) and isinstance(stats2[key], (int, float)):
            print(f"{key:<10} {stats1[key]:<15.4f} {stats2[key]:<15.4f}")
        else:
            print(f"{key:<10} {stats1[key]:<15} {stats2[key]:<15}")
    
    # Visualize the comparison
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(data1, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Normal Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    plt.hist(data2, bins=30, alpha=0.7, color='salmon', edgecolor='black')
    plt.title('Exponential Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('distribution_comparison.png')
    print("Saved comparison histograms to 'distribution_comparison.png'")

def example_5_zero_variance_case():
    """
    Example 5: Handling the special case of zero variance (constant data)
    """
    print("\nExample 5: Zero Variance Case")
    print("-------------------------")
    
    # Create a dataset with constant values
    data = [5, 5, 5, 5, 5]
    
    # Calculate descriptive statistics
    stats = calculate_descriptive_stats(data)
    
    # Print the results
    print(f"Dataset with constant values: {data}")
    print("\nDescriptive Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

def example_6_error_handling():
    """
    Example 6: Demonstrating error handling
    """
    print("\nExample 6: Error Handling")
    print("---------------------")
    
    # Example 6.1: Empty data
    print("\nAttempting to use empty data:")
    try:
        stats = calculate_descriptive_stats([])
        print("  Result:", stats)
    except Exception as e:
        print(f"  Error: {str(e)}")
    
    # Example 6.2: Non-numeric data
    print("\nAttempting to use non-numeric data:")
    try:
        stats = calculate_descriptive_stats(['a', 'b', 'c'])
        print("  Result:", stats)
    except Exception as e:
        print(f"  Error: {str(e)}")
    
    # Example 6.3: Multi-dimensional data
    print("\nAttempting to use multi-dimensional data:")
    try:
        stats = calculate_descriptive_stats([[1, 2], [3, 4]])
        print("  Result:", stats)
    except Exception as e:
        print(f"  Error: {str(e)}")
    
    # Example 6.4: Invalid nan_policy
    print("\nAttempting to use invalid nan_policy:")
    try:
        stats = calculate_descriptive_stats([1, 2, 3], nan_policy='invalid')
        print("  Result:", stats)
    except Exception as e:
        print(f"  Error: {str(e)}")

def example_7_real_world_application():
    """
    Example 7: Real-world application with pandas DataFrame
    """
    print("\nExample 7: Real-world Application")
    print("-----------------------------")
    
    # Create a sample dataset simulating temperature readings
    np.random.seed(42)  # For reproducibility
    dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
    temperatures = np.random.normal(loc=20, scale=5, size=30)  # Mean of 20°C with 5°C standard deviation
    
    # Introduce some missing values
    temperatures[5] = np.nan
    temperatures[15] = np.nan
    
    # Create a pandas DataFrame
    df = pd.DataFrame({'date': dates, 'temperature': temperatures})
    
    print("Sample temperature dataset:")
    print(df.head())
    print(f"Shape: {df.shape}")
    print(f"Missing values: {df['temperature'].isna().sum()}")
    
    # Calculate descriptive statistics using our function
    stats = calculate_descriptive_stats(df['temperature'].values, nan_policy='omit')
    
    # Print the results
    print("\nDescriptive Statistics (using our function):")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Compare with pandas describe
    print("\nComparison with pandas describe():")
    pandas_stats = df['temperature'].describe()
    print(pandas_stats)
    
    # Visualize the data
    plt.figure(figsize=(10, 6))
    plt.plot(df['date'], df['temperature'], marker='o', linestyle='-', color='blue')
    plt.axhline(stats['mean'], color='red', linestyle='dashed', label=f"Mean: {stats['mean']:.2f}°C")
    plt.fill_between(df['date'], 
                     stats['mean'] - stats['stdev'], 
                     stats['mean'] + stats['stdev'], 
                     color='red', alpha=0.2, 
                     label=f"±1 StdDev: {stats['stdev']:.2f}°C")
    plt.title('Daily Temperature Readings')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('temperature_analysis.png')
    print("Saved temperature analysis plot to 'temperature_analysis.png'")

if __name__ == "__main__":
    print("Descriptive Statistics Calculator - Usage Examples")
    print("================================================")
    
    example_1_basic_usage()
    example_2_numpy_array()
    example_3_handling_nan_values()
    example_4_comparing_datasets()
    example_5_zero_variance_case()
    example_6_error_handling()
    example_7_real_world_application()
    
    print("\nAll examples completed successfully!")
