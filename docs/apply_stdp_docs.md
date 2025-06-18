"""
STDP Implementation Documentation

This document provides detailed information about the `apply_stdp` function, which implements
Spike-Timing-Dependent Plasticity (STDP) rules for the FUM (Fully Unified Model) project.

## Overview

Spike-Timing-Dependent Plasticity (STDP) is a biological process that adjusts the strength of 
connections between neurons in the brain based on the relative timing of a particular neuron's 
output and input action potentials (or spikes). The `apply_stdp` function implements this 
mechanism for both excitatory and inhibitory synapses, with support for eligibility traces 
and parameter heterogeneity.

## Function Signature

```python
def apply_stdp(
    spike_times_pre: Union[List[float], np.ndarray],
    spike_times_post: Union[List[float], np.ndarray],
    current_weight: float,
    is_inhibitory: bool = False,
    A_plus_base: float = 0.1,
    A_minus_base: float = 0.12,
    tau_plus: float = 20.0,
    tau_minus: float = 20.0,
    eligibility_trace: float = 0.0,
    gamma: float = 0.9,
    cluster_reward: float = 0.0,
    max_reward: float = 1.0,
    spike_rate_pre: float = 0.0,
    dt: float = 1.0,
    target_rate: float = 0.3,
    eta: float = 1.0,
    A_plus_inh: Optional[float] = None,
    A_minus_inh: Optional[float] = None,
    tau_plus_inh: Optional[float] = None,
    tau_minus_inh: Optional[float] = None,
    weight_bounds: Optional[Tuple[float, float]] = None
) -> Tuple[float, float]:
```

## Parameters

### Required Parameters

- **spike_times_pre** : List[float] or np.ndarray
  - List or array of spike times for the pre-synaptic neuron (in ms).
  - Example: `[10.0, 20.0, 30.0]` represents spikes at 10ms, 20ms, and 30ms.

- **spike_times_post** : List[float] or np.ndarray
  - List or array of spike times for the post-synaptic neuron (in ms).
  - Example: `[15.0, 25.0, 35.0]` represents spikes at 15ms, 25ms, and 35ms.

- **current_weight** : float
  - The current synaptic weight (w_ij) between the neurons.
  - For excitatory synapses: positive values (e.g., 0.5)
  - For inhibitory synapses: negative values (e.g., -0.5)

### Optional Parameters

- **is_inhibitory** : bool, default=False
  - Flag indicating whether the synapse is inhibitory (True) or excitatory (False).
  - Determines which STDP rule to apply (standard or reversed timing dependency).

- **A_plus_base** : float, default=0.1
  - Base potentiation strength for excitatory synapses.
  - Typical range: 0.05 to 0.15

- **A_minus_base** : float, default=0.12
  - Base depression strength for excitatory synapses.
  - Typically slightly larger than A_plus_base to ensure stability.

- **tau_plus** : float, default=20.0
  - Time constant for potentiation (in ms) for excitatory synapses.
  - Typical range: 15ms to 25ms

- **tau_minus** : float, default=20.0
  - Time constant for depression (in ms) for excitatory synapses.
  - Typical range: 15ms to 25ms

- **eligibility_trace** : float, default=0.0
  - The current eligibility trace for the synapse.
  - Represents the memory of recent spike-timing-dependent events.

- **gamma** : float, default=0.9
  - Decay factor for the eligibility trace.
  - Range: 0 to 1, where higher values result in longer-lasting traces.

- **cluster_reward** : float, default=0.0
  - The reward signal specific to the post-synaptic neuron's cluster.
  - Range: 0 to max_reward

- **max_reward** : float, default=1.0
  - The maximum possible reward value.
  - Used to normalize cluster_reward.

- **spike_rate_pre** : float, default=0.0
  - The recent firing rate of the pre-synaptic neuron (in Hz).
  - Used for homeostatic regulation.

- **dt** : float, default=1.0
  - The time step in ms.
  - Used for updating the eligibility trace.

- **target_rate** : float, default=0.3
  - Target firing rate for homeostatic regulation (in Hz).
  - Typical value: 0.3 Hz

- **eta** : float, default=1.0
  - Base learning rate.
  - Controls the overall magnitude of weight changes.

- **A_plus_inh** : float, optional
  - Potentiation strength for inhibitory synapses.
  - If None, uses A_plus_base.

- **A_minus_inh** : float, optional
  - Depression strength for inhibitory synapses.
  - If None, uses A_minus_base.

- **tau_plus_inh** : float, optional
  - Time constant for potentiation (in ms) for inhibitory synapses.
  - If None, uses tau_plus.

- **tau_minus_inh** : float, optional
  - Time constant for depression (in ms) for inhibitory synapses.
  - If None, uses tau_minus.

- **weight_bounds** : Tuple[float, float], optional
  - Minimum and maximum allowed values for the synaptic weight.
  - If None, uses (0.0, 1.0) for excitatory and (-1.0, 0.0) for inhibitory synapses.

## Return Value

The function returns a tuple containing:

1. **new_weight** : float
   - The updated synaptic weight (w_ij) after applying STDP rules.
   - Constrained by weight_bounds.

2. **new_eligibility_trace** : float
   - The updated eligibility trace (e_ij) after applying the decay and adding the weight change.

## STDP Rules

### Excitatory Synapses (is_inhibitory=False)

For excitatory synapses, the standard STDP rule is applied:

- **Potentiation** (when pre-synaptic spike precedes post-synaptic spike, Δt > 0):
  ```
  Δw_ij = A_+ * exp(-Δt / τ_+)
  ```

- **Depression** (when post-synaptic spike precedes pre-synaptic spike, Δt < 0):
  ```
  Δw_ij = A_- * exp(Δt / τ_-)
  ```

Where:
- Δt = t_post - t_pre (the time difference between post and pre-synaptic spikes)
- A_+ is modulated by cluster reward and pre-synaptic firing rate:
  ```
  A_+ = A_plus_base * (cluster_reward / max_reward) * (spike_rate_pre / target_rate)
  ```

### Inhibitory Synapses (is_inhibitory=True)

For inhibitory synapses, the timing dependency is reversed:

- **Potentiation** (when post-synaptic spike precedes pre-synaptic spike, Δt < 0):
  ```
  Δw_ij = A_+_inh * exp(Δt / τ_+_inh)
  ```

- **Depression** (when pre-synaptic spike precedes post-synaptic spike, Δt > 0):
  ```
  Δw_ij = A_-_inh * exp(-Δt / τ_-_inh)
  ```

## Eligibility Trace

The eligibility trace is updated according to:

```
e_ij(t+dt) = gamma * e_ij(t) + Δw_ij
```

Where:
- gamma is the decay factor (0 to 1)
- Δw_ij is the weight change calculated from the STDP rules

## SIE Modulation

The learning rate is modulated by the global reward signal from the SIE:

```
Δw_ij = eta_effective * Δw_ij
```

Where eta_effective is the base learning rate (eta) potentially modulated by other factors.

## Weight Constraints

The final weight is constrained by weight_bounds:

```
new_weight = clip(current_weight + delta_w, weight_bounds[0], weight_bounds[1])
```

## Error Handling

The function performs extensive input validation to ensure all parameters have valid types and values:

- Checks that spike times are provided as lists or numpy arrays
- Ensures all time constants (tau_plus, tau_minus) are positive
- Verifies that gamma is between 0 and 1
- Confirms that cluster_reward does not exceed max_reward
- Validates that weights are consistent with the inhibitory flag (negative for inhibitory, positive for excitatory)
- Ensures weight_bounds are properly formatted and logical

## Examples

### Example 1: Excitatory Synapse

```python
import numpy as np
from apply_stdp import apply_stdp

# Define spike times
spike_times_pre = [10.0, 20.0, 30.0]  # Pre-synaptic spikes at 10ms, 20ms, 30ms
spike_times_post = [15.0, 25.0, 35.0]  # Post-synaptic spikes at 15ms, 25ms, 35ms

# Initial weight and parameters
current_weight = 0.5
cluster_reward = 0.8
max_reward = 1.0
spike_rate_pre = 0.25  # Hz

# Apply STDP
new_weight, new_trace = apply_stdp(
    spike_times_pre=spike_times_pre,
    spike_times_post=spike_times_post,
    current_weight=current_weight,
    is_inhibitory=False,
    cluster_reward=cluster_reward,
    max_reward=max_reward,
    spike_rate_pre=spike_rate_pre
)

print(f"Initial weight: {current_weight}")
print(f"New weight: {new_weight}")
print(f"Eligibility trace: {new_trace}")
```

### Example 2: Inhibitory Synapse

```python
import numpy as np
from apply_stdp import apply_stdp

# Define spike times
spike_times_pre = [10.0, 20.0, 30.0]
spike_times_post = [5.0, 15.0, 25.0]  # Post spikes precede pre spikes

# Initial weight and parameters
current_weight = -0.5  # Negative weight for inhibitory synapse
A_plus_inh = 0.08
A_minus_inh = 0.10
tau_plus_inh = 18.0
tau_minus_inh = 18.0

# Apply STDP
new_weight, new_trace = apply_stdp(
    spike_times_pre=spike_times_pre,
    spike_times_post=spike_times_post,
    current_weight=current_weight,
    is_inhibitory=True,
    A_plus_inh=A_plus_inh,
    A_minus_inh=A_minus_inh,
    tau_plus_inh=tau_plus_inh,
    tau_minus_inh=tau_minus_inh
)

print(f"Initial weight: {current_weight}")
print(f"New weight: {new_weight}")
print(f"Eligibility trace: {new_trace}")
```

### Example 3: With Eligibility Trace and Custom Weight Bounds

```python
import numpy as np
from apply_stdp import apply_stdp

# Define spike times
spike_times_pre = np.array([10.0, 20.0, 30.0, 40.0])
spike_times_post = np.array([12.0, 22.0, 32.0, 42.0])

# Initial weight and parameters
current_weight = 0.3
eligibility_trace = 0.05  # Non-zero initial eligibility trace
gamma = 0.8  # Decay factor
weight_bounds = (0.0, 0.8)  # Custom weight bounds

# Apply STDP
new_weight, new_trace = apply_stdp(
    spike_times_pre=spike_times_pre,
    spike_times_post=spike_times_post,
    current_weight=current_weight,
    eligibility_trace=eligibility_trace,
    gamma=gamma,
    weight_bounds=weight_bounds
)

print(f"Initial weight: {current_weight}")
print(f"Initial eligibility trace: {eligibility_trace}")
print(f"New weight: {new_weight}")
print(f"New eligibility trace: {new_trace}")
```

## Best Practices

1. **Spike Time Units**: Ensure all spike times are in the same unit (milliseconds).

2. **Weight Initialization**: Initialize weights appropriately:
   - Excitatory synapses: positive values (typically 0.0 to 1.0)
   - Inhibitory synapses: negative values (typically -1.0 to 0.0)

3. **Parameter Tuning**:
   - A_plus should typically be slightly smaller than A_minus for stability
   - tau_plus and tau_minus are typically in the range of 15-25ms
   - gamma controls the memory of the system; higher values (closer to 1) result in longer memory

4. **Computational Efficiency**:
   - For large numbers of spikes, consider using a time window to limit the spike pairs considered
   - The current implementation processes all possible spike pairs, which may be computationally intensive for long recordings

5. **Reward Signals**:
   - cluster_reward should be normalized relative to max_reward
   - For no reward modulation, set cluster_reward = max_reward

6. **Homeostatic Regulation**:
   - The target_rate parameter controls the homeostatic regulation
   - If spike_rate_pre > target_rate, potentiation is enhanced
   - If spike_rate_pre < target_rate, potentiation is reduced
"""
