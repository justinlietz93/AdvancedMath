from sympy import expand, factor, simplify, subs
from typing import Any, Dict

def manipulate_expression(
    expr: Any, 
    operation: str, 
    substitutions: Dict[Any, Any] = None
) -> Any:
    """
    Performs symbolic manipulation on a SymPy expression.

    Parameters
    ----------
    expr : Any
        A SymPy expression.
    operation : str
        The manipulation to perform. One of 'expand', 'factor', 'simplify', 'subs'.
    substitutions : Dict[Any, Any], optional
        A dictionary of substitutions to make, required for the 'subs' operation, 
        by default None.

    Returns
    -------
    Any
        The manipulated SymPy expression.
    """
    if operation == 'expand':
        return expand(expr)
    elif operation == 'factor':
        return factor(expr)
    elif operation == 'simplify':
        return simplify(expr)
    elif operation == 'subs':
        if substitutions is None:
            raise ValueError("Substitutions must be provided for the 'subs' operation.")
        return expr.subs(substitutions)
    else:
        raise ValueError(f"Unsupported operation: {operation}")
