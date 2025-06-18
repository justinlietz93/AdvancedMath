from sympy import solve, Eq
from typing import Any, List

def solve_equation(
    equation: Any, 
    variable: Any
) -> List[Any]:
    """
    Symbolically solves an equation for a given variable.

    Parameters
    ----------
    equation : Any
        A SymPy Eq object or an expression that is assumed to be equal to zero.
    variable : Any
        The SymPy symbol to solve for.

    Returns
    -------
    List[Any]
        A list of solutions for the variable.
    """
    return solve(equation, variable)
