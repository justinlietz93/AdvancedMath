import numpy as np
from scipy.stats import wasserstein_distance
from typing import Union

def calculate_wasserstein_distance(
    u_values: np.ndarray,
    v_values: np.ndarray,
    u_weights: np.ndarray = None,
    v_weights: np.ndarray = None
) -> float:
    """
    Calculates the 1-D Wasserstein distance between two distributions.

    Parameters
    ----------
    u_values : np.ndarray
        A 1D array of values for the first distribution.
    v_values : np.ndarray
        A 1D array of values for the second distribution.
    u_weights : np.ndarray, optional
        Weights for the first distribution, by default None.
    v_weights : np.ndarray, optional
        Weights for the second distribution, by default None.

    Returns
    -------
    float
        The 1-D Wasserstein distance.
    """
    return wasserstein_distance(u_values, v_values, u_weights, v_weights)
