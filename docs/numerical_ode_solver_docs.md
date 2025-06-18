# Numerical ODE Solver Tool Documentation

## Overview

The `numerical_ode_solver` function provides a robust interface for numerically solving ordinary differential equation (ODE) initial value problems. It leverages SciPy's `solve_ivp` function to compute solutions to systems of first-order ODEs.

## Purpose

This tool is designed to solve systems of first-order ODEs of the form:

```
dy/dt = f(t, y, *args)
```

with initial conditions:

```
y(t0) = y0
```

It is particularly useful for simulating dynamic systems like Leaky Integrate-and-Fire (LIF) neuron models and other time-dependent phenomena.

## Requirements

- Python 3.6+
- NumPy
- SciPy

## Function Signature

```python
def numerical_ode_solver(
    fun: Callable,
    t_span: Tuple[float, float],
    y0: Union[List[float], np.ndarray],
    t_eval: Optional[Union[List[float], np.ndarray]] = None,
    args: Tuple = (),
    method: str = 'RK45',
    rtol: float = 1e-3,
    atol: float = 1e-6,
    max_step: Optional[float] = None
) -> Any:
```

## Parameters

### Required Parameters

- **fun** : callable
  - Function that defines the ODE system
  - Calling signature: `fun(t, y, *args)`
  - Must return an array-like with the same shape as `y`
  - Example: `def my_ode(t, y, param1, param2): return [-param1 * y[0], param2 * y[1]]`

- **t_span** : tuple of float
  - Interval of integration `(t0, tf)`
  - The solver starts with `t=t0` and integrates until it reaches `t=tf`
  - Example: `(0, 10)` to solve from t=0 to t=10

- **y0** : array_like
  - Initial state
  - Must be a 1-D array or list of floats
  - Example: `[1.0, 0.0]` for a system with two variables

### Optional Parameters

- **t_eval** : array_like or None, optional
  - Times at which to store the computed solution
  - If None (default), the solver will choose the time points automatically
  - Example: `np.linspace(0, 10, 100)` for 100 evenly spaced points

- **args** : tuple, optional
  - Extra arguments to pass to the function `fun`
  - Default is an empty tuple
  - Example: `(0.1, 0.2)` to pass two parameters to the ODE function

- **method** : str, optional
  - Integration method to use
  - Default: 'RK45' (Explicit Runge-Kutta method of order 5(4))
  - Options:
    - 'RK45': Explicit Runge-Kutta method of order 5(4)
    - 'RK23': Explicit Runge-Kutta method of order 3(2)
    - 'DOP853': Explicit Runge-Kutta method of order 8
    - 'Radau': Implicit Runge-Kutta method of the Radau IIA family of order 5
    - 'BDF': Implicit multi-step variable-order (1 to 5) method
    - 'LSODA': Adams/BDF method with automatic stiffness detection

- **rtol** : float, optional
  - Relative tolerance for the solver
  - Default: 1e-3
  - Controls the relative error of the solution

- **atol** : float, optional
  - Absolute tolerance for the solver
  - Default: 1e-6
  - Controls the absolute error of the solution

- **max_step** : float or None, optional
  - Maximum allowed step size
  - If None (default), the solver will determine it automatically
  - Useful for controlling the resolution of the solution

## Return Value

- **sol** : OdeResult
  - Object with the following attributes:
    - `t`: array, times at which the solution was computed
    - `y`: array, values of the solution at corresponding times in `t`
    - `sol`: callable, interpolated solution
    - `t_events`, `y_events`: arrays (only if events were detected)
    - `nfev`, `njev`, `nlu`: number of evaluations of the right-hand side, Jacobian, LU decompositions
    - `status`: int, reason for algorithm termination
    - `message`: str, human-readable description of the termination reason
    - `success`: bool, whether the solver succeeded

## Exceptions

- **ValueError**
  - Raised if input parameters are invalid:
    - If `t_span` is not a 2-element tuple
    - If `t_span[0] >= t_span[1]` (start time must be less than end time)
    - If `y0` is not array-like or not 1-dimensional
    - If `method` is not recognized
    - If `rtol` or `atol` are not positive numbers
    - If the ODE function returns an array of incorrect shape

- **TypeError**
  - Raised if input parameters have incorrect types:
    - If `fun` is not callable
    - If `y0` cannot be converted to a float array
    - If `t_eval` cannot be converted to a float array
    - If `args` is not a tuple

- **RuntimeError**
  - Raised if the solver fails to converge or encounters other runtime issues:
    - If `solve_ivp` returns with `success=False`
    - If the solver exceeds maximum number of steps
    - If other numerical issues occur during integration

## Usage Examples

### Example 1: Simple Exponential Decay

```python
import numpy as np
import matplotlib.pyplot as plt
from numerical_ode_solver import numerical_ode_solver

# Define the ODE: dy/dt = -k*y
def exponential_decay(t, y, rate_constant):
    return -rate_constant * y

# Set parameters
t_span = (0, 10)  # Time span from 0 to 10
y0 = [1.0]        # Initial condition y(0) = 1
rate_constant = 0.1  # Decay rate

# Solve the ODE
sol = numerical_ode_solver(exponential_decay, t_span, y0, args=(rate_constant,))

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(sol.t, sol.y[0], 'b-', linewidth=2)
plt.grid(True)
plt.xlabel('Time', fontsize=12)
plt.ylabel('y(t)', fontsize=12)
plt.title('Exponential Decay: dy/dt = -0.1y', fontsize=14)
plt.savefig('exponential_decay.png')
plt.show()
```

### Example 2: Lotka-Volterra Predator-Prey Model

```python
import numpy as np
import matplotlib.pyplot as plt
from numerical_ode_solver import numerical_ode_solver

# Define the Lotka-Volterra model
def lotka_volterra(t, z, a, b, c, d):
    x, y = z  # x: prey population, y: predator population
    dx_dt = a * x - b * x * y  # Rate of change of prey population
    dy_dt = -c * y + d * x * y  # Rate of change of predator population
    return [dx_dt, dy_dt]

# Set parameters
t_span = (0, 15)  # Time span
y0 = [10, 5]      # Initial populations: 10 prey, 5 predators
params = (1.5, 1, 3, 1)  # a, b, c, d parameters
t_eval = np.linspace(0, 15, 1000)  # Specific evaluation points

# Solve the ODE
sol = numerical_ode_solver(lotka_volterra, t_span, y0, t_eval, args=params)

# Plot the results
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(sol.t, sol.y[0], 'g-', label='Prey')
plt.plot(sol.t, sol.y[1], 'r-', label='Predator')
plt.grid(True)
plt.legend()
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Lotka-Volterra Model: Population vs Time')

plt.subplot(2, 1, 2)
plt.plot(sol.y[0], sol.y[1], 'b-')
plt.grid(True)
plt.xlabel('Prey Population')
plt.ylabel('Predator Population')
plt.title('Lotka-Volterra Model: Phase Space')
plt.tight_layout()
plt.savefig('lotka_volterra.png')
plt.show()
```

### Example 3: Harmonic Oscillator

```python
import numpy as np
import matplotlib.pyplot as plt
from numerical_ode_solver import numerical_ode_solver

# Define the harmonic oscillator
def harmonic_oscillator(t, y, omega):
    """
    Harmonic oscillator ODE: d²x/dt² + omega²*x = 0
    Rewritten as a system of first-order ODEs:
    dx/dt = v
    dv/dt = -omega²*x
    """
    x, v = y
    dxdt = v
    dvdt = -(omega**2) * x
    return [dxdt, dvdt]

# Set parameters
omega = 2.0  # Angular frequency
t_span = (0, 10)
y0 = [1.0, 0.0]  # Initial position and velocity
t_eval = np.linspace(0, 10, 500)

# Solve the ODE
sol = numerical_ode_solver(harmonic_oscillator, t_span, y0, t_eval, args=(omega,))

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], 'b-', label='Position')
plt.plot(sol.t, sol.y[1], 'r-', label='Velocity')
plt.grid(True)
plt.legend()
plt.xlabel('Time')
plt.ylabel('Value')
plt.title(f'Harmonic Oscillator (ω = {omega})')
plt.savefig('harmonic_oscillator.png')
plt.show()
```

## Best Practices

1. **Function Definition**: Ensure your ODE function follows the required signature `fun(t, y, *args)` and returns an array of the same shape as `y`.

2. **Initial Conditions**: Provide initial conditions `y0` as a 1D array or list, with one value for each variable in your system.

3. **Time Span**: Set `t_span` as a tuple `(t0, tf)` where `t0` is the start time and `tf` is the end time.

4. **Method Selection**:
   - For non-stiff problems, use 'RK45' (default) or 'DOP853'
   - For stiff problems, use 'Radau', 'BDF', or 'LSODA'
   - If unsure, start with 'LSODA' which automatically switches between methods

5. **Tolerance Settings**: Adjust `rtol` and `atol` to control accuracy:
   - Decrease for higher accuracy (e.g., `rtol=1e-6, atol=1e-9`)
   - Increase for faster computation with lower accuracy

6. **Error Handling**: Always check the `success` attribute of the returned solution to verify the solver completed successfully.

7. **Accessing Results**: The solution's `y` attribute is a 2D array where:
   - `sol.y[i]` is the solution for the i-th variable across all time points
   - `sol.y[:, j]` is the solution for all variables at the j-th time point

8. **Interpolation**: Use the `sol` attribute of the result to interpolate the solution at any time within the integration interval:
   ```python
   t_interp = 2.5  # Time point within t_span
   y_interp = sol.sol(t_interp)  # Interpolated solution at t=2.5
   ```

## Common Issues and Solutions

1. **Solver Fails to Converge**:
   - Try a different method (e.g., switch to 'BDF' for stiff problems)
   - Decrease `max_step` to force smaller steps
   - Adjust tolerances (`rtol` and `atol`)

2. **Slow Computation**:
   - Use a less demanding method (e.g., 'RK23' instead of 'RK45')
   - Increase tolerance values
   - Limit the number of output points with `t_eval`

3. **Inaccurate Results**:
   - Decrease tolerance values
   - Use a higher-order method (e.g., 'DOP853')
   - Ensure the ODE function is correctly implemented

4. **Discontinuities in Solution**:
   - For systems with discontinuities, consider breaking the integration into separate intervals
   - Use the `events` parameter of `solve_ivp` (accessible through additional parameters)
