import numpy as np
from typing import Dict
from ripser import ripser

def compute_persistent_homology(
    points: np.ndarray, 
    max_dim: int = 1
) -> Dict[str, np.ndarray]:
    """
    Computes the persistent homology of a point cloud using Vietoris-Rips filtration.

    Parameters
    ----------
    points : np.ndarray
        A numpy array of shape (n_points, n_features) representing the data,
        or a distance matrix of shape (n_points, n_points).
    max_dim : int, optional
        The maximum dimension of homology to compute, by default 1.

    Returns
    -------
    Dict[str, np.ndarray]
        A dictionary containing the persistence diagrams for each dimension,
        as well as the distance matrix used. The keys are 'dgms' and 'dists'.
    """
    if not isinstance(points, np.ndarray) or points.ndim != 2:
        raise TypeError("points must be a 2D numpy array.")
    if not isinstance(max_dim, int) or max_dim < 0:
        raise ValueError("max_dim must be a non-negative integer.")

    # If the input is a square matrix, assume it's a distance matrix
    if points.shape[0] == points.shape[1]:
        result = ripser(points, maxdim=max_dim, distance_matrix=True)
    else:
        result = ripser(points, maxdim=max_dim)
    
    return result
