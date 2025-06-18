import numpy as np
import matplotlib.pyplot as plt
from numerical_integrate import numerical_integrate

def test_numerical_integrate():
    """Test the numerical_integrate function with various integration scenarios."""
    print("Testing numerical_integrate function...")
    
    # Test case 1: Simple polynomial (x^2 from 0 to 1, analytical result = 1/3)
    result, error = numerical_integrate(lambda x: x**2, 0, 1)
    analytical = 1/3
    print(f"Test case 1 (x^2 from 0 to 1):")
    print(f"  Result: {result:.8f}, Error: {error:.8e}")
    print(f"  Analytical: {analytical:.8f}")
    print(f"  Absolute difference: {abs(result - analytical):.8e}")
    
    # Test case 2: Trigonometric function (sin(x) from 0 to pi, analytical result = 2)
    result, error = numerical_integrate(lambda x: np.sin(x), 0, np.pi)
    analytical = 2.0
    print(f"\nTest case 2 (sin(x) from 0 to pi):")
    print(f"  Result: {result:.8f}, Error: {error:.8e}")
    print(f"  Analytical: {analytical:.8f}")
    print(f"  Absolute difference: {abs(result - analytical):.8e}")
    
    # Test case 3: Exponential function (e^(-x^2) from -inf to inf, analytical result = sqrt(pi))
    result, error = numerical_integrate(lambda x: np.exp(-x**2), -np.inf, np.inf)
    analytical = np.sqrt(np.pi)
    print(f"\nTest case 3 (e^(-x^2) from -inf to inf):")
    print(f"  Result: {result:.8f}, Error: {error:.8e}")
    print(f"  Analytical: {analytical:.8f}")
    print(f"  Absolute difference: {abs(result - analytical):.8e}")
    
    # Test case 4: Function with additional parameters
    def integrand(x, a, b):
        return a * np.sin(b * x)
    
    result, error = numerical_integrate(integrand, 0, np.pi, args=(2.0, 1.0))
    analytical = 4.0
    print(f"\nTest case 4 (2*sin(x) from 0 to pi):")
    print(f"  Result: {result:.8f}, Error: {error:.8e}")
    print(f"  Analytical: {analytical:.8f}")
    print(f"  Absolute difference: {abs(result - analytical):.8e}")
    
    # Test case 5: Oscillatory function (sin(10x) from 0 to 2pi)
    result, error = numerical_integrate(lambda x: np.sin(10*x), 0, 2*np.pi)
    analytical = 0.0  # Integral of sin(10x) over a complete period is 0
    print(f"\nTest case 5 (sin(10x) from 0 to 2pi):")
    print(f"  Result: {result:.8f}, Error: {error:.8e}")
    print(f"  Analytical: {analytical:.8f}")
    print(f"  Absolute difference: {abs(result - analytical):.8e}")
    
    # Test case 6: Error handling for invalid inputs
    print("\nTest case 6 (Error handling):")
    try:
        numerical_integrate("not_a_function", 0, 1)
        print("  Failed: TypeError not raised for non-callable function")
    except TypeError:
        print("  Passed: TypeError raised for non-callable function")
    
    try:
        numerical_integrate(lambda x: x**2, 1, 0)
        print("  Failed: ValueError not raised for a > b")
    except ValueError:
        print("  Passed: ValueError raised for a > b")
    
    # Test case 7: Function with singularity (handled by quad)
    print("\nTest case 7 (Function with singularity):")
    try:
        result, error = numerical_integrate(lambda x: 1/np.sqrt(x), 0, 1)
        print(f"  Result: {result:.8f}, Error: {error:.8e}")
        analytical = 2.0
        print(f"  Analytical: {analytical:.8f}")
        print(f"  Absolute difference: {abs(result - analytical):.8e}")
    except RuntimeError as e:
        print(f"  Failed: {str(e)}")

def plot_integration_examples():
    """Create plots to visualize some integration examples."""
    # Example 1: Polynomial
    x = np.linspace(0, 1, 100)
    y = x**2
    
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    plt.plot(x, y)
    plt.fill_between(x, y, alpha=0.3)
    plt.title('∫x² dx from 0 to 1')
    plt.grid(True, alpha=0.3)
    
    # Example 2: Sine function
    x = np.linspace(0, np.pi, 100)
    y = np.sin(x)
    
    plt.subplot(2, 2, 2)
    plt.plot(x, y)
    plt.fill_between(x, y, alpha=0.3)
    plt.title('∫sin(x) dx from 0 to π')
    plt.grid(True, alpha=0.3)
    
    # Example 3: Gaussian function
    x = np.linspace(-3, 3, 100)
    y = np.exp(-x**2)
    
    plt.subplot(2, 2, 3)
    plt.plot(x, y)
    plt.fill_between(x, y, alpha=0.3)
    plt.title('∫e^(-x²) dx from -∞ to ∞')
    plt.grid(True, alpha=0.3)
    
    # Example 4: Function with parameters
    x = np.linspace(0, np.pi, 100)
    y = 2 * np.sin(x)
    
    plt.subplot(2, 2, 4)
    plt.plot(x, y)
    plt.fill_between(x, y, alpha=0.3)
    plt.title('∫2·sin(x) dx from 0 to π')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('integration_examples.png')
    print("\nIntegration examples plot saved as 'integration_examples.png'")

if __name__ == "__main__":
    test_numerical_integrate()
    plot_integration_examples()
