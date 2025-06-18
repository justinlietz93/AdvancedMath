import numpy as np
from .information_theory import calculate_mutual_information

def information_bottleneck(
    p_xy: np.ndarray,
    p_xt: np.ndarray,
    beta: float
) -> float:
    """
    Calculates the Information Bottleneck objective function.

    L(T) = I(X;T) - β * I(X;Y)

    Note: This function calculates the objective for a given set of distributions.
    The optimization of this objective is a separate, more complex task.

    Parameters
    ----------
    p_xy : np.ndarray
        The joint probability distribution P(X, Y).
    p_xt : np.ndarray
        The joint probability distribution P(X, T), where T is the compressed
        representation of X.
    beta : float
        The Lagrange multiplier that balances compression and information
        preservation.

    Returns
    -------
    float
        The value of the Information Bottleneck objective function.
    """
    i_xt = calculate_mutual_information(p_xt)
    i_xy = calculate_mutual_information(p_xy)
    
    return i_xt - beta * i_xy
