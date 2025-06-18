import numpy as np
from linear_system_solver import linear_system_solver

def example_1_simple_2x2():
    """
    Example 1: Simple 2x2 system
    3x + 2y = 7
    x + y = 3
    
    Expected solution: x = 1, y = 2
    """
    print("\nExample 1: Simple 2x2 system")
    print("3x + 2y = 7")
    print("x + y = 3")
    
    # Define the coefficient matrix and right-hand side vector
    A = np.array([[3, 2], [1, 1]])
    b = np.array([7, 3])
    
    # Solve the system
    x = linear_system_solver(A, b)
    
    print(f"Solution: x = {x[0]}, y = {x[1]}")
    print(f"Verification: {np.allclose(np.dot(A, x), b)}")

def example_2_3x3_system():
    """
    Example 2: 3x3 system
    2x + y + z = 5
    3x + 5y + 2z = 15
    x + 3y + 5z = 14
    
    Expected solution: x = 1, y = 2, z = 3
    """
    print("\nExample 2: 3x3 system")
    print("2x + y + z = 5")
    print("3x + 5y + 2z = 15")
    print("x + 3y + 5z = 14")
    
    # Define the coefficient matrix and right-hand side vector
    A = np.array([[2, 1, 1], [3, 5, 2], [1, 3, 5]])
    b = np.array([5, 15, 14])
    
    # Solve the system
    x = linear_system_solver(A, b)
    
    print(f"Solution: x = {x[0]}, y = {x[1]}, z = {x[2]}")
    print(f"Verification: {np.allclose(np.dot(A, x), b)}")

def example_3_list_inputs():
    """
    Example 3: Using list inputs instead of NumPy arrays
    4x + 2y = 10
    2x + 5y = 16
    
    Expected solution: x = 2, y = 2
    """
    print("\nExample 3: Using list inputs")
    print("4x + 2y = 10")
    print("2x + 5y = 16")
    
    # Define the coefficient matrix and right-hand side vector as lists
    A = [[4, 2], [2, 5]]
    b = [10, 16]
    
    # Solve the system
    x = linear_system_solver(A, b)
    
    print(f"Solution: x = {x[0]}, y = {x[1]}")
    print(f"Verification: {np.allclose(np.dot(A, x), b)}")

def example_4_engineering_application():
    """
    Example 4: Engineering application - Circuit analysis
    
    Consider a circuit with three loops and three unknown currents I1, I2, I3.
    Using Kirchhoff's voltage law, we get the following system:
    
    10I1 - 4I2 - 0I3 = 12 (Loop 1)
    -4I1 + 8I2 - 2I3 = 0  (Loop 2)
    0I1 - 2I2 + 6I3 = 6   (Loop 3)
    
    Expected solution: I1, I2, I3 in amperes
    """
    print("\nExample 4: Engineering application - Circuit analysis")
    print("10I1 - 4I2 - 0I3 = 12 (Loop 1)")
    print("-4I1 + 8I2 - 2I3 = 0  (Loop 2)")
    print("0I1 - 2I2 + 6I3 = 6   (Loop 3)")
    
    # Define the coefficient matrix and right-hand side vector
    A = np.array([[10, -4, 0], [-4, 8, -2], [0, -2, 6]])
    b = np.array([12, 0, 6])
    
    # Solve the system
    x = linear_system_solver(A, b)
    
    print(f"Solution: I1 = {x[0]:.4f} A, I2 = {x[1]:.4f} A, I3 = {x[2]:.4f} A")
    print(f"Verification: {np.allclose(np.dot(A, x), b)}")

def example_5_error_handling():
    """
    Example 5: Demonstrating error handling
    """
    print("\nExample 5: Demonstrating error handling")
    
    # Example 5.1: Non-square matrix
    try:
        print("\nAttempting to solve with non-square matrix:")
        A = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3 matrix
        b = np.array([7, 8])
        x = linear_system_solver(A, b)
    except ValueError as e:
        print(f"Caught error: {e}")
    
    # Example 5.2: Incompatible dimensions
    try:
        print("\nAttempting to solve with incompatible dimensions:")
        A = np.array([[1, 2], [3, 4]])  # 2x2 matrix
        b = np.array([5, 6, 7])  # 3-element vector
        x = linear_system_solver(A, b)
    except ValueError as e:
        print(f"Caught error: {e}")
    
    # Example 5.3: Singular matrix
    try:
        print("\nAttempting to solve with singular matrix:")
        A = np.array([[1, 2], [2, 4]])  # Singular matrix (row 2 = 2*row 1)
        b = np.array([3, 6])
        x = linear_system_solver(A, b)
    except np.linalg.LinAlgError as e:
        print(f"Caught error: {e}")
    
    # Example 5.4: Non-numeric data
    try:
        print("\nAttempting to solve with non-numeric data:")
        A = np.array([['a', 'b'], ['c', 'd']])
        b = np.array(['e', 'f'])
        x = linear_system_solver(A, b)
    except TypeError as e:
        print(f"Caught error: {e}")

if __name__ == "__main__":
    print("Linear System Solver - Usage Examples")
    print("=====================================")
    
    example_1_simple_2x2()
    example_2_3x3_system()
    example_3_list_inputs()
    example_4_engineering_application()
    example_5_error_handling()
