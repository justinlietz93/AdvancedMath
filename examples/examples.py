"""
Examples demonstrating the usage of the apply_stdp function.

This script contains various examples showing how to use the apply_stdp function
for different scenarios including excitatory and inhibitory synapses, with
different parameter configurations.
"""

import numpy as np
import matplotlib.pyplot as plt
from archive.apply_stdp import apply_stdp

def example_1_basic_excitatory():
    """
    Example 1: Basic usage with excitatory synapse.
    
    This example demonstrates the effect of STDP on an excitatory synapse
    where post-synaptic spikes consistently follow pre-synaptic spikes,
    leading to potentiation (strengthening).
    """
    print("\n=== Example 1: Basic Excitatory Synapse ===")
    
    # Define spike times
    spike_times_pre = [10.0, 20.0, 30.0]  # Pre-synaptic spikes at 10ms, 20ms, 30ms
    spike_times_post = [15.0, 25.0, 35.0]  # Post-synaptic spikes at 15ms, 25ms, 35ms
    
    # Initial weight and parameters
    current_weight = 0.5
    
    # Apply STDP
    new_weight, new_trace = apply_stdp(
        spike_times_pre=spike_times_pre,
        spike_times_post=spike_times_post,
        current_weight=current_weight,
        is_inhibitory=False
    )
    
    # Print results
    print(f"Initial weight: {current_weight}")
    print(f"New weight: {new_weight:.6f}")
    print(f"Eligibility trace: {new_trace:.6f}")
    print(f"Weight change: {new_weight - current_weight:.6f}")
    
    # Visualize the spike timing
    plt.figure(figsize=(10, 4))
    plt.scatter(spike_times_pre, np.ones_like(spike_times_pre), color='blue', marker='|', s=100, label='Pre-synaptic')
    plt.scatter(spike_times_post, np.ones_like(spike_times_post) * 1.5, color='red', marker='|', s=100, label='Post-synaptic')
    
    for pre, post in zip(spike_times_pre, spike_times_post):
        plt.annotate('', xy=(post, 1.5), xytext=(pre, 1), 
                    arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
        plt.text((pre + post) / 2, 1.25, f'Δt={post-pre}ms', ha='center', va='center', color='green')
    
    plt.yticks([1, 1.5], ['Pre-synaptic', 'Post-synaptic'])
    plt.xlabel('Time (ms)')
    plt.title('Spike Timing - Excitatory Synapse (Potentiation)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('example_1_excitatory.png')
    print("Plot saved as 'example_1_excitatory.png'")


def example_2_basic_inhibitory():
    """
    Example 2: Basic usage with inhibitory synapse.
    
    This example demonstrates the effect of STDP on an inhibitory synapse
    where post-synaptic spikes consistently precede pre-synaptic spikes,
    leading to potentiation (strengthening of inhibition).
    """
    print("\n=== Example 2: Basic Inhibitory Synapse ===")
    
    # Define spike times
    spike_times_pre = [10.0, 20.0, 30.0]  # Pre-synaptic spikes at 10ms, 20ms, 30ms
    spike_times_post = [5.0, 15.0, 25.0]  # Post-synaptic spikes at 5ms, 15ms, 25ms
    
    # Initial weight and parameters
    current_weight = -0.5  # Negative weight for inhibitory synapse
    
    # Apply STDP
    new_weight, new_trace = apply_stdp(
        spike_times_pre=spike_times_pre,
        spike_times_post=spike_times_post,
        current_weight=current_weight,
        is_inhibitory=True
    )
    
    # Print results
    print(f"Initial weight: {current_weight}")
    print(f"New weight: {new_weight:.6f}")
    print(f"Eligibility trace: {new_trace:.6f}")
    print(f"Weight change: {new_weight - current_weight:.6f}")
    
    # Visualize the spike timing
    plt.figure(figsize=(10, 4))
    plt.scatter(spike_times_pre, np.ones_like(spike_times_pre), color='blue', marker='|', s=100, label='Pre-synaptic')
    plt.scatter(spike_times_post, np.ones_like(spike_times_post) * 1.5, color='red', marker='|', s=100, label='Post-synaptic')
    
    for pre, post in zip(spike_times_pre, spike_times_post):
        plt.annotate('', xy=(pre, 1), xytext=(post, 1.5), 
                    arrowprops=dict(arrowstyle='->', color='purple', lw=1.5))
        plt.text((pre + post) / 2, 1.25, f'Δt={post-pre}ms', ha='center', va='center', color='purple')
    
    plt.yticks([1, 1.5], ['Pre-synaptic', 'Post-synaptic'])
    plt.xlabel('Time (ms)')
    plt.title('Spike Timing - Inhibitory Synapse (Potentiation)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('example_2_inhibitory.png')
    print("Plot saved as 'example_2_inhibitory.png'")


def example_3_reward_modulation():
    """
    Example 3: Reward modulation.
    
    This example demonstrates how cluster reward affects the STDP learning
    for excitatory synapses.
    """
    print("\n=== Example 3: Reward Modulation ===")
    
    # Define spike times
    spike_times_pre = [10.0, 20.0, 30.0, 40.0]
    spike_times_post = [15.0, 25.0, 35.0, 45.0]
    
    # Initial weight
    current_weight = 0.5
    
    # Different reward levels
    reward_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = []
    
    for reward in reward_levels:
        # Apply STDP with different reward levels
        new_weight, new_trace = apply_stdp(
            spike_times_pre=spike_times_pre,
            spike_times_post=spike_times_post,
            current_weight=current_weight,
            cluster_reward=reward,
            max_reward=1.0
        )
        
        results.append((reward, new_weight, new_trace))
        print(f"Reward: {reward:.2f}, New weight: {new_weight:.6f}, Weight change: {new_weight - current_weight:.6f}")
    
    # Visualize the effect of reward on weight change
    plt.figure(figsize=(10, 6))
    rewards = [r[0] for r in results]
    weights = [r[1] for r in results]
    weight_changes = [w - current_weight for w in weights]
    
    plt.plot(rewards, weights, 'o-', color='blue', label='New Weight')
    plt.axhline(y=current_weight, color='gray', linestyle='--', label='Initial Weight')
    
    plt.xlabel('Cluster Reward')
    plt.ylabel('Synaptic Weight')
    plt.title('Effect of Reward on STDP Learning')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('example_3_reward_modulation.png')
    print("Plot saved as 'example_3_reward_modulation.png'")


def example_4_eligibility_trace():
    """
    Example 4: Eligibility trace.
    
    This example demonstrates how the eligibility trace affects learning
    over multiple applications of STDP.
    """
    print("\n=== Example 4: Eligibility Trace ===")
    
    # Define spike times for multiple time steps
    spike_patterns = [
        # Time step 1: Strong pre-post correlation
        ([10.0, 20.0, 30.0], [15.0, 25.0, 35.0]),
        # Time step 2: No spikes
        ([], []),
        # Time step 3: No spikes
        ([], []),
        # Time step 4: Weak pre-post correlation
        ([10.0], [15.0])
    ]
    
    # Initial parameters
    current_weight = 0.5
    eligibility_trace = 0.0
    gamma_values = [0.0, 0.5, 0.9]  # Different decay factors
    
    # Track results for different gamma values
    all_weights = {gamma: [current_weight] for gamma in gamma_values}
    all_traces = {gamma: [eligibility_trace] for gamma in gamma_values}
    
    # Run simulation for each gamma value
    for gamma in gamma_values:
        weight = current_weight
        trace = eligibility_trace
        
        print(f"\nSimulation with gamma = {gamma}:")
        print(f"Initial - Weight: {weight:.6f}, Trace: {trace:.6f}")
        
        # Apply STDP for each time step
        for step, (pre_spikes, post_spikes) in enumerate(spike_patterns, 1):
            weight, trace = apply_stdp(
                spike_times_pre=pre_spikes,
                spike_times_post=post_spikes,
                current_weight=weight,
                eligibility_trace=trace,
                gamma=gamma
            )
            
            all_weights[gamma].append(weight)
            all_traces[gamma].append(trace)
            
            print(f"Step {step} - Weight: {weight:.6f}, Trace: {trace:.6f}")
    
    # Visualize the effect of eligibility trace
    plt.figure(figsize=(12, 8))
    
    # Plot weights
    plt.subplot(2, 1, 1)
    time_steps = range(len(spike_patterns) + 1)
    for gamma, weights in all_weights.items():
        plt.plot(time_steps, weights, 'o-', label=f'γ = {gamma}')
    
    plt.xlabel('Time Step')
    plt.ylabel('Synaptic Weight')
    plt.title('Effect of Eligibility Trace on Weight Evolution')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot traces
    plt.subplot(2, 1, 2)
    for gamma, traces in all_traces.items():
        plt.plot(time_steps, traces, 'o-', label=f'γ = {gamma}')
    
    plt.xlabel('Time Step')
    plt.ylabel('Eligibility Trace')
    plt.title('Evolution of Eligibility Trace')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('example_4_eligibility_trace.png')
    print("Plot saved as 'example_4_eligibility_trace.png'")


def example_5_stdp_curve():
    """
    Example 5: STDP curve.
    
    This example visualizes the STDP curve showing weight change as a function
    of the time difference between pre- and post-synaptic spikes.
    """
    print("\n=== Example 5: STDP Curve ===")
    
    # Parameters
    A_plus = 0.1
    A_minus = 0.12
    tau_plus = 20.0
    tau_minus = 20.0
    
    A_plus_inh = 0.1
    A_minus_inh = 0.12
    tau_plus_inh = 20.0
    tau_minus_inh = 20.0
    
    # Generate time differences
    delta_t = np.linspace(-100, 100, 1000)  # From -100ms to 100ms
    
    # Calculate weight changes for excitatory synapses
    delta_w_exc = np.zeros_like(delta_t)
    for i, dt in enumerate(delta_t):
        if dt > 0:  # Post after pre (potentiation)
            delta_w_exc[i] = A_plus * np.exp(-dt / tau_plus)
        else:  # Pre after post (depression)
            delta_w_exc[i] = -A_minus * np.exp(dt / tau_minus)
    
    # Calculate weight changes for inhibitory synapses
    delta_w_inh = np.zeros_like(delta_t)
    for i, dt in enumerate(delta_t):
        if dt < 0:  # Post before pre (potentiation for inhibitory)
            delta_w_inh[i] = A_plus_inh * np.exp(dt / tau_plus_inh)
        else:  # Pre before post (depression for inhibitory)
            delta_w_inh[i] = -A_minus_inh * np.exp(-dt / tau_minus_inh)
    
    # Visualize the STDP curves
    plt.figure(figsize=(12, 6))
    
    plt.plot(delta_t, delta_w_exc, 'b-', label='Excitatory Synapse')
    plt.plot(delta_t, delta_w_inh, 'r-', label='Inhibitory Synapse')
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    plt.xlabel('Δt = t_post - t_pre (ms)')
    plt.ylabel('Weight Change (Δw)')
    plt.title('STDP Curve: Weight Change vs Spike Timing Difference')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('example_5_stdp_curve.png')
    print("Plot saved as 'example_5_stdp_curve.png'")


def example_6_parameter_variation():
    """
    Example 6: Parameter variation.
    
    This example demonstrates how varying STDP parameters affects the learning.
    """
    print("\n=== Example 6: Parameter Variation ===")
    
    # Define spike times
    spike_times_pre = [10.0, 20.0, 30.0]
    spike_times_post = [15.0, 25.0, 35.0]
    
    # Initial weight
    current_weight = 0.5
    
    # Vary tau_plus
    tau_values = [10.0, 20.0, 30.0, 40.0]
    results_tau = []
    
    print("\nVarying tau_plus:")
    for tau in tau_values:
        new_weight, new_trace = apply_stdp(
            spike_times_pre=spike_times_pre,
            spike_times_post=spike_times_post,
            current_weight=current_weight,
            tau_plus=tau
        )
        
        results_tau.append((tau, new_weight))
        print(f"tau_plus: {tau:.1f}, New weight: {new_weight:.6f}, Weight change: {new_weight - current_weight:.6f}")
    
    # Vary A_plus
    A_values = [0.05, 0.1, 0.15, 0.2]
    results_A = []
    
    print("\nVarying A_plus_base:")
    for A in A_values:
        new_weight, new_trace = apply_stdp(
            spike_times_pre=spike_times_pre,
            spike_times_post=spike_times_post,
            current_weight=current_weight,
            A_plus_base=A
        )
        
        results_A.append((A, new_weight))
        print(f"A_plus_base: {A:.2f}, New weight: {new_weight:.6f}, Weight change: {new_weight - current_weight:.6f}")
    
    # Visualize the effect of parameter variation
    plt.figure(figsize=(12, 6))
    
    # Plot effect of tau_plus
    plt.subplot(1, 2, 1)
    taus = [r[0] for r in results_tau]
    weights_tau = [r[1] for r in results_tau]
    
    plt.plot(taus, weights_tau, 'o-', color='blue')
    plt.axhline(y=current_weight, color='gray', linestyle='--', label='Initial Weight')
    
    plt.xlabel('tau_plus (ms)')
    plt.ylabel('New Synaptic Weight')
    plt.title('Effect of tau_plus on STDP Learning')
    plt.grid(True, alpha=0.3)
    
    # Plot effect of A_plus_base
    plt.subplot(1, 2, 2)
    As = [r[0] for r in results_A]
    weights_A = [r[1] for r in results_A]
    
    plt.plot(As, weights_A, 'o-', color='green')
    plt.axhline(y=current_weight, color='gray', linestyle='--', label='Initial Weight')
    
    plt.xlabel('A_plus_base')
    plt.ylabel('New Synaptic Weight')
    plt.title('Effect of A_plus_base on STDP Learning')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('example_6_parameter_variation.png')
    print("Plot saved as 'example_6_parameter_variation.png'")


def example_7_error_handling():
    """
    Example 7: Error handling.
    
    This example demonstrates how the apply_stdp function handles various errors.
    """
    print("\n=== Example 7: Error Handling ===")
    
    # Define valid inputs
    spike_times_pre = [10.0, 20.0, 30.0]
    spike_times_post = [15.0, 25.0, 35.0]
    current_weight = 0.5
    
    # Test case 1: Invalid spike_times_pre type
    print("\nTest case 1: Invalid spike_times_pre type")
    try:
        apply_stdp(
            spike_times_pre="not a list",
            spike_times_post=spike_times_post,
            current_weight=current_weight
        )
    except Exception as e:
        print(f"Caught exception as expected: {type(e).__name__}: {str(e)}")
    
    # Test case 2: Invalid current_weight type
    print("\nTest case 2: Invalid current_weight type")
    try:
        apply_stdp(
            spike_times_pre=spike_times_pre,
            spike_times_post=spike_times_post,
            current_weight="not a number"
        )
    except Exception as e:
        print(f"Caught exception as expected: {type(e).__name__}: {str(e)}")
    
    # Test case 3: Negative time constant
    print("\nTest case 3: Negative time constant")
    try:
        apply_stdp(
            spike_times_pre=spike_times_pre,
            spike_times_post=spike_times_post,
            current_weight=current_weight,
            tau_plus=-10.0
        )
    except Exception as e:
        print(f"Caught exception as expected: {type(e).__name__}: {str(e)}")
    
    # Test case 4: Inconsistent weight and inhibitory flag
    print("\nTest case 4: Inconsistent weight and inhibitory flag")
    try:
        apply_stdp(
            spike_times_pre=spike_times_pre,
            spike_times_post=spike_times_post,
            current_weight=0.5,  # Positive weight
            is_inhibitory=True   # Inhibitory flag
        )
    except Exception as e:
        print(f"Caught exception as expected: {type(e).__name__}: {str(e)}")
    
    # Test case 5: Invalid gamma value
    print("\nTest case 5: Invalid gamma value")
    try:
        apply_stdp(
            spike_times_pre=spike_times_pre,
            spike_times_post=spike_times_post,
            current_weight=current_weight,
            gamma=1.5  # gamma must be between 0 and 1
        )
    except Exception as e:
        print(f"Caught exception as expected: {type(e).__name__}: {str(e)}")


if __name__ == "__main__":
    print("STDP Implementation - Usage Examples")
    print("====================================")
    
    # Run all examples
    example_1_basic_excitatory()
    example_2_basic_inhibitory()
    example_3_reward_modulation()
    example_4_eligibility_trace()
    example_5_stdp_curve()
    example_6_parameter_variation()
    example_7_error_handling()
    
    print("\nAll examples completed!")
