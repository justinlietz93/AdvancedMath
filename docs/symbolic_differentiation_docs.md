# Symbolic Differentiation Tool Documentation

## Overview

The `symbolic_differentiate` function provides a robust interface for performing symbolic differentiation of mathematical expressions. It leverages the SymPy library to compute derivatives of expressions with respect to one or more variables.

## Features

- Accepts expressions as either strings or SymPy expression objects
- Supports differentiation with respect to a single variable or multiple variables (partial derivatives)
- Handles higher-order derivatives
- Provides comprehensive error handling and input validation
- Returns results as SymPy expression objects for further manipulation

## Requirements

- Python 3.6+
- SymPy library (`pip install sympy`)

## Function Signature

```python
def symbolic_differentiate(
    expression: Union[str, sympy.Expr], 
    variable: Union[str, sympy.Symbol, List[Union[str, sympy.Symbol]]], 
    order: int = 1
) -> sympy.Expr:
```

## Parameters

- **expression**: The mathematical expression to differentiate
  - Type: `str` or `sympy.Expr`
  - If a string is provided, it will be parsed into a SymPy expression
  - Examples: `"x**2 + 2*x + 1"`, `x**2 + sin(y)`

- **variable**: The variable(s) with respect to which to differentiate
  - Type: `str`, `sympy.Symbol`, or a list of these
  - For a single variable: `"x"` or `Symbol('x')`
  - For multiple variables (partial derivatives): `["x", "y"]` or `[Symbol('x'), Symbol('y')]`

- **order**: The order of the derivative (default: 1)
  - Type: `int`
  - Must be a positive integer
  - Example: `2` for second derivative

## Return Value

- A SymPy expression representing the derivative
- Type: `sympy.Expr`

## Exceptions

- **TypeError**: Raised when input types are incorrect
- **ValueError**: Raised when:
  - The order is not a positive integer
  - A variable list is empty
  - A variable in a multi-variable list is not present in the expression
- **sympy.SympifyError**: Raised when a string expression cannot be parsed

## Special Cases

- When differentiating with respect to a variable not present in the expression:
  - For a single variable: Returns 0 (following calculus rules)
  - For multiple variables: Raises ValueError if any variable is not in the expression

## Usage Examples

### Basic Usage

```python
from symbolic_differentiation import symbolic_differentiate

# Differentiate a polynomial
result = symbolic_differentiate("x**2 + 2*x + 1", "x")
print(result)  # Output: 2*x + 2

# Differentiate a trigonometric expression
result = symbolic_differentiate("sin(x) + cos(x)", "x")
print(result)  # Output: cos(x) - sin(x)
```

### Using SymPy Objects Directly

```python
from sympy import symbols, sin, cos, exp
from symbolic_differentiation import symbolic_differentiate

x, y = symbols('x y')
expression = x**2 * sin(y) + exp(x)

# Differentiate with respect to x
result = symbolic_differentiate(expression, x)
print(result)  # Output: 2*x*sin(y) + exp(x)

# Differentiate with respect to y
result = symbolic_differentiate(expression, y)
print(result)  # Output: x**2*cos(y)
```

### Higher-Order Derivatives

```python
from symbolic_differentiation import symbolic_differentiate

# Second derivative of a polynomial
result = symbolic_differentiate("x**3 + x**2 + x + 1", "x", order=2)
print(result)  # Output: 6*x + 2
```

### Partial Derivatives

```python
from symbolic_differentiation import symbolic_differentiate

# Partial derivative with respect to x, then y
result = symbolic_differentiate("x**2 * y**3", ["x", "y"])
print(result)  # Output: 6*x*y**2

# Equivalent to:
result1 = symbolic_differentiate("x**2 * y**3", "x")  # 2*x*y**3
result2 = symbolic_differentiate(result1, "y")        # 6*x*y**2
```

### Error Handling Examples

```python
from symbolic_differentiation import symbolic_differentiate

try:
    # Invalid expression string
    symbolic_differentiate("x^2 + 1", "x")  # ^ is not valid in SymPy expressions
except sympy.SympifyError as e:
    print(f"Expression parsing error: {e}")

try:
    # Invalid order
    symbolic_differentiate("x**2", "x", order=0)
except ValueError as e:
    print(f"Value error: {e}")  # Order must be a positive integer

try:
    # Empty variable list
    symbolic_differentiate("x**2", [])
except ValueError as e:
    print(f"Value error: {e}")  # Variable list cannot be empty

try:
    # Multiple variables with one not in expression
    symbolic_differentiate("x**2", ["x", "z"])
except ValueError as e:
    print(f"Value error: {e}")  # Variable 'z' is not present in the expression
```

## Best Practices

1. **Input Validation**: While the function handles input validation internally, it's good practice to ensure your expressions and variables are valid before calling the function.

2. **Performance**: For complex expressions or repeated calculations, create SymPy expression objects once and reuse them rather than passing strings repeatedly.

3. **Result Manipulation**: The returned SymPy expression can be further manipulated using SymPy's functions:
   ```python
   from sympy import simplify
   result = symbolic_differentiate("sin(x)**2 + cos(x)**2", "x")
   simplified = simplify(result)  # Simplifies to 0
   ```

4. **Numerical Evaluation**: To evaluate the derivative at specific points:
   ```python
   from sympy import symbols, N
   x = symbols('x')
   derivative = symbolic_differentiate("x**2", x)
   value_at_3 = N(derivative.subs(x, 3))  # Evaluates to 6
   ```

5. **Integration with Other Tools**: The returned SymPy expressions can be used with other SymPy functions or converted to other formats:
   ```python
   from sympy import latex
   derivative = symbolic_differentiate("sin(x)", "x")
   latex_representation = latex(derivative)  # For use in documents
   ```
