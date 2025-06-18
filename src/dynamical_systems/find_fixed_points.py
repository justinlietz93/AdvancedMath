import numpy as np
from scipy.optimize import fsolve
from typing import Callable, List, Any

def find_fixed_points(
    func: Callable[[np.ndarray, Any], np.ndarray],
    initial_guesses: List[np.ndarray]
) -> np.ndarray:
    """
    Finds the fixed points (equilibria) of a dynamical system.

    Parameters
    ----------
    func : Callable[[np.ndarray, Any], np.ndarray]
        A function representing the dynamical system, where func(y, *args) = dy/dt.
    initial_guesses : List[np.ndarray]
        A list of initial guesses for the fixed points.

    Returns
    -------
    np.ndarray
        An array of the found fixed points.
    """
    fixed_points = []
    for guess in initial_guesses:
        fixed_point, _, _, _ = fsolve(func, guess, full_output=True)
        fixed_points.append(fixed_point)
    return np.array(fixed_points)
