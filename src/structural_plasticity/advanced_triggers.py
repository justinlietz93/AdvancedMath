import numpy as np

def calculate_advanced_growth_trigger(
    avg_reward: float,
    burst_score: float,
    bdnf_proxy: float,
    kappa: float = 2.0,
    nu: float = 0.8,
    rho: float = 0.5
) -> float:
    """
    Calculates an advanced, biologically-inspired growth trigger.

    G(c,t) = σ(κ * (avg_reward[c] - 0.5) + ν * burst_score[c] + ρ * bdnf_proxy[c])

    Parameters
    ----------
    avg_reward : float
        The average reward of the cluster.
    burst_score : float
        The burst score of the cluster.
    bdnf_proxy : float
        The BDNF proxy level of the cluster.
    kappa : float, optional
    nu : float, optional
    rho : float, optional

    Returns
    -------
    float
        The calculated growth trigger value.
    """
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    arg = kappa * (avg_reward - 0.5) + nu * burst_score + rho * bdnf_proxy
    return sigmoid(arg)
