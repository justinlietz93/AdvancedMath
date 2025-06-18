import numpy as np
import time
import matplotlib.pyplot as plt
from apply_stdp import apply_stdp as revised_apply_stdp

# Import the original function for comparison
import sys
sys.path.append('/home/ubuntu')
from archive.apply_stdp import apply_stdp as original_apply_stdp

def test_functionality_equivalence():
    """Test that the revised function produces the same results as the original."""
    print("Testing functionality equivalence...")
    
    # Test case 1: Excitatory synapse with potentiation
    spike_times_pre = [10.0, 20.0, 30.0]
    spike_times_post = [15.0, 25.0, 35.0]
    current_weight = 0.2
    
    # Run original function
    orig_weight, orig_trace = original_apply_stdp(
        spike_times_pre=spike_times_pre,
        spike_times_post=spike_times_post,
        current_weight=current_weight,
        is_inhibitory=False
    )
    
    # Run revised function
    rev_weight, rev_trace = revised_apply_stdp(
        spike_times_pre=spike_times_pre,
        spike_times_post=spike_times_post,
        current_weight=current_weight,
        is_inhibitory=False
    )
    
    print(f"Test case 1 (Excitatory potentiation):")
    print(f"  Original: weight={orig_weight:.6f}, trace={orig_trace:.6f}")
    print(f"  Revised:  weight={rev_weight:.6f}, trace={rev_trace:.6f}")
    
    # Test case 2: Excitatory synapse with depression
    spike_times_pre = [15.0, 25.0, 35.0]
    spike_times_post = [10.0, 20.0, 30.0]
    current_weight = 0.8
    
    # Run original function
    orig_weight, orig_trace = original_apply_stdp(
        spike_times_pre=spike_times_pre,
        spike_times_post=spike_times_post,
        current_weight=current_weight,
        is_inhibitory=False
    )
    
    # Run revised function
    rev_weight, rev_trace = revised_apply_stdp(
        spike_times_pre=spike_times_pre,
        spike_times_post=spike_times_post,
        current_weight=current_weight,
        is_inhibitory=False
    )
    
    print(f"Test case 2 (Excitatory depression):")
    print(f"  Original: weight={orig_weight:.6f}, trace={orig_trace:.6f}")
    print(f"  Revised:  weight={rev_weight:.6f}, trace={rev_trace:.6f}")
    
    # Test case 3: Inhibitory synapse with potentiation
    spike_times_pre = [15.0, 25.0, 35.0]
    spike_times_post = [10.0, 20.0, 30.0]
    current_weight = -0.5
    
    # Run original function
    orig_weight, orig_trace = original_apply_stdp(
        spike_times_pre=spike_times_pre,
        spike_times_post=spike_times_post,
        current_weight=current_weight,
        is_inhibitory=True
    )
    
    # Run revised function
    rev_weight, rev_trace = revised_apply_stdp(
        spike_times_pre=spike_times_pre,
        spike_times_post=spike_times_post,
        current_weight=current_weight,
        is_inhibitory=True
    )
    
    print(f"Test case 3 (Inhibitory potentiation):")
    print(f"  Original: weight={orig_weight:.6f}, trace={orig_trace:.6f}")
    print(f"  Revised:  weight={rev_weight:.6f}, trace={rev_trace:.6f}")
    
    # Test case 4: Inhibitory synapse with depression
    spike_times_pre = [10.0, 20.0, 30.0]
    spike_times_post = [15.0, 25.0, 35.0]
    current_weight = -0.5
    
    # Run original function
    orig_weight, orig_trace = original_apply_stdp(
        spike_times_pre=spike_times_pre,
        spike_times_post=spike_times_post,
        current_weight=current_weight,
        is_inhibitory=True
    )
    
    # Run revised function
    rev_weight, rev_trace = revised_apply_stdp(
        spike_times_pre=spike_times_pre,
        spike_times_post=spike_times_post,
        current_weight=current_weight,
        is_inhibitory=True
    )
    
    print(f"Test case 4 (Inhibitory depression):")
    print(f"  Original: weight={orig_weight:.6f}, trace={orig_trace:.6f}")
    print(f"  Revised:  weight={rev_weight:.6f}, trace={rev_trace:.6f}")
    
    # Test case 5: With reward modulation
    spike_times_pre = [10.0, 20.0, 30.0]
    spike_times_post = [15.0, 25.0, 35.0]
    current_weight = 0.5
    
    # Run original function
    orig_weight, orig_trace = original_apply_stdp(
        spike_times_pre=spike_times_pre,
        spike_times_post=spike_times_post,
        current_weight=current_weight,
        cluster_reward=0.8,
        max_reward=1.0
    )
    
    # Run revised function
    rev_weight, rev_trace = revised_apply_stdp(
        spike_times_pre=spike_times_pre,
        spike_times_post=spike_times_post,
        current_weight=current_weight,
        cluster_reward=0.8,
        max_reward=1.0
    )
    
    print(f"Test case 5 (With reward modulation):")
    print(f"  Original: weight={orig_weight:.6f}, trace={orig_trace:.6f}")
    print(f"  Revised:  weight={rev_weight:.6f}, trace={rev_trace:.6f}")
    
    # Test case 6: With eligibility trace
    spike_times_pre = [10.0, 20.0, 30.0]
    spike_times_post = [15.0, 25.0, 35.0]
    current_weight = 0.5
    eligibility_trace = 0.1
    gamma = 0.8
    
    # Run original function
    orig_weight, orig_trace = original_apply_stdp(
        spike_times_pre=spike_times_pre,
        spike_times_post=spike_times_post,
        current_weight=current_weight,
        eligibility_trace=eligibility_trace,
        gamma=gamma
    )
    
    # Run revised function
    rev_weight, rev_trace = revised_apply_stdp(
        spike_times_pre=spike_times_pre,
        spike_times_post=spike_times_post,
        current_weight=current_weight,
        eligibility_trace=eligibility_trace,
        gamma=gamma
    )
    
    print(f"Test case 6 (With eligibility trace):")
    print(f"  Original: weight={orig_weight:.6f}, trace={orig_trace:.6f}")
    print(f"  Revised:  weight={rev_weight:.6f}, trace={rev_trace:.6f}")

def test_performance():
    """Test the performance improvement of the revised function."""
    print("\nTesting performance improvement...")
    
    # Create larger spike trains for performance testing
    np.random.seed(42)  # For reproducibility
    n_spikes = 1000
    
    # Generate random spike times
    spike_times_pre = np.sort(np.random.uniform(0, 1000, n_spikes))
    spike_times_post = np.sort(np.random.uniform(0, 1000, n_spikes))
    current_weight = 0.5
    
    # Time the original function
    start_time = time.time()
    original_apply_stdp(
        spike_times_pre=spike_times_pre,
        spike_times_post=spike_times_post,
        current_weight=current_weight
    )
    original_time = time.time() - start_time
    
    # Time the revised function
    start_time = time.time()
    revised_apply_stdp(
        spike_times_pre=spike_times_pre,
        spike_times_post=spike_times_post,
        current_weight=current_weight
    )
    revised_time = time.time() - start_time
    
    print(f"Performance with {n_spikes} spikes:")
    print(f"  Original function time: {original_time:.6f} seconds")
    print(f"  Revised function time:  {revised_time:.6f} seconds")
    print(f"  Speedup factor: {original_time / revised_time:.2f}x")
    
    # Test with different spike counts
    spike_counts = [10, 100, 500, 1000, 2000]
    original_times = []
    revised_times = []
    
    for n in spike_counts:
        # Generate random spike times
        spike_times_pre = np.sort(np.random.uniform(0, 1000, n))
        spike_times_post = np.sort(np.random.uniform(0, 1000, n))
        
        # Time the original function
        start_time = time.time()
        original_apply_stdp(
            spike_times_pre=spike_times_pre,
            spike_times_post=spike_times_post,
            current_weight=current_weight
        )
        original_times.append(time.time() - start_time)
        
        # Time the revised function
        start_time = time.time()
        revised_apply_stdp(
            spike_times_pre=spike_times_pre,
            spike_times_post=spike_times_post,
            current_weight=current_weight
        )
        revised_times.append(time.time() - start_time)
    
    # Plot performance comparison
    plt.figure(figsize=(10, 6))
    plt.plot(spike_counts, original_times, 'o-', label='Original Function')
    plt.plot(spike_counts, revised_times, 'o-', label='Revised Function')
    plt.xlabel('Number of Spikes')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Performance Comparison: Original vs. Revised apply_stdp')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('performance_comparison.png')
    print("Performance comparison plot saved as 'performance_comparison.png'")

if __name__ == "__main__":
    test_functionality_equivalence()
    test_performance()
