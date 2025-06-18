import numpy as np
from typing import Dict, Any

def analyze_stability(
    jacobian: np.ndarray
) -> Dict[str, Any]:
    """
    Analyzes the stability of a fixed point by examining the eigenvalues of the Jacobian.

    Parameters
    ----------
    jacobian : np.ndarray
        The Jacobian matrix at the fixed point.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the stability analysis, including:
        - 'eigenvalues': The eigenvalues of the Jacobian.
        - 'stability_type': The type of stability (e.g., stable node, saddle point).
    """
    eigenvalues = np.linalg.eigvals(jacobian)
    
    real_parts = np.real(eigenvalues)
    imag_parts = np.imag(eigenvalues)

    if np.all(real_parts < 0):
        if np.all(imag_parts == 0):
            stability_type = "Stable Node"
        else:
            stability_type = "Stable Spiral"
    elif np.all(real_parts > 0):
        if np.all(imag_parts == 0):
            stability_type = "Unstable Node"
        else:
            stability_type = "Unstable Spiral"
    elif np.any(real_parts > 0) and np.any(real_parts < 0):
        stability_type = "Saddle Point"
    else:
        stability_type = "Center (Marginally Stable)"

    return {
        'eigenvalues': eigenvalues,
        'stability_type': stability_type
    }
