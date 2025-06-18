# Advanced Math Toolkit for AI and Neuroscience

This repository contains a collection of Python tools for advanced mathematics, with a focus on applications in artificial intelligence, computational neuroscience, and complex systems modeling. These tools are designed to be used by researchers, developers, and AI agents (like LLMs) to build and analyze sophisticated models like the Fully Unified Model (FUM).

## Core Mathematical Concepts

This toolkit is designed to support the exploration and implementation of several novel, high-level mathematical frameworks for building adaptive, emergent intelligence. The key concepts include:

*   **TDA-Dynamical Hybrid for Cluster-Based Reward:** A framework for modeling the complex, multi-level interactions between learning clusters as a system of coupled oscillators whose dynamics are constrained by the topology of the network. This unifies micro-level interactions with macro-level emergent behavior.
*   **Stochastic-Topological Bifurcation Theory:** A method for understanding and predicting large-scale qualitative shifts in the FUM's behavior by modeling the system's state as a persistence diagram and detecting phase transitions as fundamental changes in its topology.
*   **Lyapunov-Game Hybrid for Reward Dynamics:** An elegant solution to the problem of reward hacking that models the different components of the Self-Improvement Engine (SIE) as agents in a potential game, with a global Lyapunov function providing a formal guarantee of stability.

## Features

This toolkit provides a comprehensive suite of modules organized into the following packages:

*   **Neuroscience (`src/neuro`):**
    *   Spike-Timing-Dependent Plasticity (STDP)
    *   Synaptic Tagging and Capture (STC)
    *   Advanced SIE reward functions

*   **Structural Plasticity (`src/structural_plasticity`):**
    *   Biologically-inspired growth triggers
    *   Burst detection
    *   Pruning and rewiring

*   **Graph Theory & Analysis (`src/graph`):**
    *   Standard graph metrics
    *   Community detection
    *   Graph edit distance and PageRank
    *   Graph dynamics simulation and path scoring
    *   Coarse-graining

*   **Topological Data Analysis (TDA) (`src/tda`):**
    *   Vietoris-Rips complex construction
    *   Persistent homology
    *   Custom TDA metrics

*   **Dynamical Systems (`src/dynamical_systems`):**
    *   Fixed point finding
    *   Jacobian calculation
    *   Stability analysis

*   **Symbolic Mathematics (`src/symbolic`):**
    *   Symbol and function definition
    *   Expression manipulation
    *   Equation solving
    *   Symbolic calculus and logic

*   **Stochastic Processes (`src/stochastic`, `src/sde`):**
    *   Gillespie simulation
    *   Stochastic Differential Equation (SDE) solver

*   **Information Theory (`src/info_theory`):**
    *   Entropy, mutual information, and KL divergence
    *   Information bottleneck analysis

*   **And more...** including toolkits for Causal Inference, Fractal Analysis, Self-Organized Criticality (SOC), Optimal Transport, and Evolutionary Algorithms.

## Getting Started

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/AdvancedMath.git
    cd AdvancedMath
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

Each tool is located in the `src/` directory and has corresponding documentation in the `docs/` directory and examples in the `examples/` directory.

For example, to use the TDA tools:

```python
import numpy as np
from src.tda import compute_persistent_homology, calculate_tda_metrics

# Generate some sample data
points = np.random.rand(100, 2)

# Compute the persistent homology
result = compute_persistent_homology(points, max_dim=1)

# Calculate the TDA metrics
metrics = calculate_tda_metrics(result['dgms'])

print(f"TDA Metrics: {metrics}")
```

## Running Tests

To run the unit tests for the toolkit, use the following command:

```bash
python -m unittest discover -s tests
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.
