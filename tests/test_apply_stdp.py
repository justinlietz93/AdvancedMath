"""
Unit tests for the apply_stdp function.

This script contains tests to verify the functionality of the apply_stdp function.
"""

import unittest
import numpy as np
from archive.apply_stdp import apply_stdp

class TestApplySTDP(unittest.TestCase):
    """Test cases for the apply_stdp function."""
    
    def test_excitatory_potentiation(self):
        """Test STDP potentiation for excitatory synapses."""
        # Pre-synaptic spikes precede post-synaptic spikes (Δt > 0)
        # This should lead to potentiation (weight increase)
        spike_times_pre = [10.0, 20.0, 30.0]
        spike_times_post = [15.0, 25.0, 35.0]
        current_weight = 0.2  # Lower initial weight to allow for potentiation
        
        new_weight, new_trace = apply_stdp(
            spike_times_pre=spike_times_pre,
            spike_times_post=spike_times_post,
            current_weight=current_weight,
            is_inhibitory=False
        )
        
        # Check that weight increased (potentiation occurred)
        self.assertGreater(new_weight, current_weight)
        # Check that weight is within bounds
        self.assertLessEqual(new_weight, 1.0)
        self.assertGreaterEqual(new_weight, 0.0)
        # Check that eligibility trace is positive
        self.assertGreater(new_trace, 0.0)
    
    def test_excitatory_depression(self):
        """Test STDP depression for excitatory synapses."""
        # Post-synaptic spikes precede pre-synaptic spikes (Δt < 0)
        # This should lead to depression (weight decrease)
        spike_times_pre = [15.0, 25.0, 35.0]
        spike_times_post = [10.0, 20.0, 30.0]
        current_weight = 0.5
        
        new_weight, new_trace = apply_stdp(
            spike_times_pre=spike_times_pre,
            spike_times_post=spike_times_post,
            current_weight=current_weight,
            is_inhibitory=False
        )
        
        # Check that weight decreased (depression occurred)
        self.assertLess(new_weight, current_weight)
        # Check that weight is within bounds
        self.assertLessEqual(new_weight, 1.0)
        self.assertGreaterEqual(new_weight, 0.0)
        # Check that eligibility trace is negative
        self.assertLess(new_trace, 0.0)
    
    def test_inhibitory_potentiation(self):
        """Test STDP potentiation for inhibitory synapses."""
        # Post-synaptic spikes precede pre-synaptic spikes (Δt < 0)
        # For inhibitory synapses, this should lead to potentiation (more negative weight)
        spike_times_pre = [15.0, 25.0, 35.0]
        spike_times_post = [10.0, 20.0, 30.0]
        current_weight = -0.5
        
        new_weight, new_trace = apply_stdp(
            spike_times_pre=spike_times_pre,
            spike_times_post=spike_times_post,
            current_weight=current_weight,
            is_inhibitory=True
        )
        
        # Check that weight became more negative (potentiation for inhibitory)
        self.assertLess(new_weight, current_weight)
        # Check that weight is within bounds
        self.assertLessEqual(new_weight, 0.0)
        self.assertGreaterEqual(new_weight, -1.0)
        # Check that eligibility trace is negative (for inhibitory potentiation)
        self.assertLess(new_trace, 0.0)
    
    def test_inhibitory_depression(self):
        """Test STDP depression for inhibitory synapses."""
        # Pre-synaptic spikes precede post-synaptic spikes (Δt > 0)
        # For inhibitory synapses, this should lead to depression (less negative weight)
        spike_times_pre = [10.0, 20.0, 30.0]
        spike_times_post = [15.0, 25.0, 35.0]
        current_weight = -0.5
        
        new_weight, new_trace = apply_stdp(
            spike_times_pre=spike_times_pre,
            spike_times_post=spike_times_post,
            current_weight=current_weight,
            is_inhibitory=True
        )
        
        # Check that weight became less negative (depression for inhibitory)
        self.assertGreater(new_weight, current_weight)
        # Check that weight is within bounds
        self.assertLessEqual(new_weight, 0.0)
        self.assertGreaterEqual(new_weight, -1.0)
        # Check that eligibility trace is positive (for inhibitory depression)
        self.assertGreater(new_trace, 0.0)
    
    def test_reward_modulation(self):
        """Test reward modulation of STDP."""
        spike_times_pre = [10.0, 20.0, 30.0]
        spike_times_post = [15.0, 25.0, 35.0]
        current_weight = 0.5
        
        # Apply STDP with no reward
        new_weight_no_reward, _ = apply_stdp(
            spike_times_pre=spike_times_pre,
            spike_times_post=spike_times_post,
            current_weight=current_weight,
            cluster_reward=0.0,
            max_reward=1.0
        )
        
        # Apply STDP with full reward
        new_weight_full_reward, _ = apply_stdp(
            spike_times_pre=spike_times_pre,
            spike_times_post=spike_times_post,
            current_weight=current_weight,
            cluster_reward=1.0,
            max_reward=1.0
        )
        
        # Check that reward increases potentiation
        self.assertGreater(new_weight_full_reward, new_weight_no_reward)
    
    def test_eligibility_trace(self):
        """Test eligibility trace integration."""
        spike_times_pre = [10.0]
        spike_times_post = [15.0]
        current_weight = 0.5
        initial_trace = 0.1
        gamma = 0.9
        
        # Apply STDP with initial eligibility trace
        new_weight, new_trace = apply_stdp(
            spike_times_pre=spike_times_pre,
            spike_times_post=spike_times_post,
            current_weight=current_weight,
            eligibility_trace=initial_trace,
            gamma=gamma
        )
        
        # Calculate expected trace based on the formula: e_ij(t+dt) = gamma * e_ij(t) + Δw_ij
        # We can't know the exact Δw_ij without duplicating the implementation,
        # but we can check that the trace decays appropriately
        self.assertGreaterEqual(new_trace, gamma * initial_trace)
        
        # Test with gamma = 0 (no memory)
        new_weight_no_memory, new_trace_no_memory = apply_stdp(
            spike_times_pre=spike_times_pre,
            spike_times_post=spike_times_post,
            current_weight=current_weight,
            eligibility_trace=initial_trace,
            gamma=0.0
        )
        
        # With gamma = 0, the trace should not include any contribution from the previous trace
        self.assertNotEqual(new_trace_no_memory, 0.0)  # Should still include current Δw_ij
        self.assertLess(abs(new_trace_no_memory), abs(new_trace))  # Should be smaller without memory
    
    def test_weight_bounds(self):
        """Test that weights are properly bounded."""
        spike_times_pre = [10.0, 20.0, 30.0]
        spike_times_post = [15.0, 25.0, 35.0]
        
        # Test excitatory synapse with custom bounds
        current_weight_exc = 0.5
        custom_bounds_exc = (0.2, 0.8)
        
        new_weight_exc, _ = apply_stdp(
            spike_times_pre=spike_times_pre,
            spike_times_post=spike_times_post,
            current_weight=current_weight_exc,
            is_inhibitory=False,
            weight_bounds=custom_bounds_exc
        )
        
        # Check that weight is within custom bounds
        self.assertLessEqual(new_weight_exc, custom_bounds_exc[1])
        self.assertGreaterEqual(new_weight_exc, custom_bounds_exc[0])
        
        # Test inhibitory synapse with custom bounds
        current_weight_inh = -0.5
        custom_bounds_inh = (-0.8, -0.2)
        
        new_weight_inh, _ = apply_stdp(
            spike_times_pre=spike_times_pre,
            spike_times_post=spike_times_post,
            current_weight=current_weight_inh,
            is_inhibitory=True,
            weight_bounds=custom_bounds_inh
        )
        
        # Check that weight is within custom bounds
        self.assertLessEqual(new_weight_inh, custom_bounds_inh[1])
        self.assertGreaterEqual(new_weight_inh, custom_bounds_inh[0])
    
    def test_learning_rate_modulation(self):
        """Test learning rate (eta) modulation."""
        spike_times_pre = [10.0, 20.0, 30.0]
        spike_times_post = [15.0, 25.0, 35.0]
        current_weight = 0.5
        
        # Apply STDP with default learning rate (eta=1.0)
        new_weight_default, _ = apply_stdp(
            spike_times_pre=spike_times_pre,
            spike_times_post=spike_times_post,
            current_weight=current_weight
        )
        
        # Apply STDP with doubled learning rate
        new_weight_doubled, _ = apply_stdp(
            spike_times_pre=spike_times_pre,
            spike_times_post=spike_times_post,
            current_weight=current_weight,
            eta=2.0
        )
        
        # Apply STDP with halved learning rate
        new_weight_halved, _ = apply_stdp(
            spike_times_pre=spike_times_pre,
            spike_times_post=spike_times_post,
            current_weight=current_weight,
            eta=0.5
        )
        
        # Check that learning rate properly scales weight changes
        weight_change_default = new_weight_default - current_weight
        weight_change_doubled = new_weight_doubled - current_weight
        weight_change_halved = new_weight_halved - current_weight
        
        # The weight change should scale approximately linearly with eta
        # Allow for some numerical precision issues
        self.assertAlmostEqual(weight_change_doubled / weight_change_default, 2.0, places=5)
        self.assertAlmostEqual(weight_change_halved / weight_change_default, 0.5, places=5)
    
    def test_empty_spike_trains(self):
        """Test behavior with empty spike trains."""
        # No spikes should result in no weight change
        current_weight = 0.5
        eligibility_trace = 0.1
        gamma = 0.9
        
        new_weight, new_trace = apply_stdp(
            spike_times_pre=[],
            spike_times_post=[],
            current_weight=current_weight,
            eligibility_trace=eligibility_trace,
            gamma=gamma
        )
        
        # Weight should remain unchanged
        self.assertEqual(new_weight, current_weight)
        # Trace should decay according to gamma
        self.assertEqual(new_trace, gamma * eligibility_trace)
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Valid inputs for reference
        valid_pre = [10.0, 20.0]
        valid_post = [15.0, 25.0]
        valid_weight_exc = 0.5
        valid_weight_inh = -0.5
        
        # Test invalid spike_times_pre type
        with self.assertRaises(TypeError):
            apply_stdp(
                spike_times_pre="not a list",
                spike_times_post=valid_post,
                current_weight=valid_weight_exc
            )
        
        # Test invalid current_weight type
        with self.assertRaises(TypeError):
            apply_stdp(
                spike_times_pre=valid_pre,
                spike_times_post=valid_post,
                current_weight="not a number"
            )
        
        # Test negative time constant
        with self.assertRaises(ValueError):
            apply_stdp(
                spike_times_pre=valid_pre,
                spike_times_post=valid_post,
                current_weight=valid_weight_exc,
                tau_plus=-10.0
            )
        
        # Test inconsistent weight and inhibitory flag
        with self.assertRaises(ValueError):
            apply_stdp(
                spike_times_pre=valid_pre,
                spike_times_post=valid_post,
                current_weight=valid_weight_exc,  # Positive weight
                is_inhibitory=True               # Inhibitory flag
            )
        
        with self.assertRaises(ValueError):
            apply_stdp(
                spike_times_pre=valid_pre,
                spike_times_post=valid_post,
                current_weight=valid_weight_inh,  # Negative weight
                is_inhibitory=False              # Excitatory flag
            )
        
        # Test invalid gamma value
        with self.assertRaises(ValueError):
            apply_stdp(
                spike_times_pre=valid_pre,
                spike_times_post=valid_post,
                current_weight=valid_weight_exc,
                gamma=1.5  # gamma must be between 0 and 1
            )
        
        # Test invalid weight_bounds
        with self.assertRaises(TypeError):
            apply_stdp(
                spike_times_pre=valid_pre,
                spike_times_post=valid_post,
                current_weight=valid_weight_exc,
                weight_bounds="not a tuple"
            )
        
        with self.assertRaises(ValueError):
            apply_stdp(
                spike_times_pre=valid_pre,
                spike_times_post=valid_post,
                current_weight=valid_weight_exc,
                weight_bounds=(0.5, 0.3)  # min > max
            )

if __name__ == "__main__":
    unittest.main(verbosity=2)
