"""Integration with scientific libraries example for jscip.

This example demonstrates:
- Using theta_sampling mode for optimization
- Converting between representations
- Integration with scipy.optimize
- Working with NumPy arrays
"""

import numpy as np
from scipy.optimize import minimize

from jscip import DerivedScalarParameter, IndependentScalarParameter, ParameterBank

# Define a simple optimization problem: minimize a quadratic function
# f(x, y) = (x - 2)^2 + (y + 1)^2
# Optimal solution: x=2, y=-1

x_param = IndependentScalarParameter(value=0.0, is_sampled=True, range=(-5.0, 5.0))
y_param = IndependentScalarParameter(value=0.0, is_sampled=True, range=(-5.0, 5.0))


def compute_objective(params):
    """Objective function to minimize"""
    return (params["x"] - 2.0) ** 2 + (params["y"] + 1.0) ** 2


objective = DerivedScalarParameter(compute_objective)

# Create bank with theta_sampling for optimization
bank = ParameterBank(
    parameters={
        "x": x_param,
        "y": y_param,
        "objective": objective,
    },
    theta_sampling=True,  # Work with flat arrays
)

print("=" * 60)
print("Integration with Scientific Libraries")
print("=" * 60)

# 1. Demonstrate theta_sampling mode
print("\n1. Theta sampling mode:")
print("   Sampling returns flat arrays instead of ParameterSet objects")

theta_sample = bank.sample()
print(f"   Sample (theta): {theta_sample}")
print(f"   Shape: {theta_sample.shape}")

# Convert back to full parameter set
full_sample = bank.theta_to_instance(theta_sample)
print(f"   Converted to ParameterSet:")
print(f"     x={full_sample['x']:.3f}, y={full_sample['y']:.3f}")
print(f"     objective={full_sample['objective']:.3f}")

# 2. Sample multiple configurations
print("\n2. Batch sampling with theta mode:")
theta_batch = bank.sample(size=5)
print(f"   Batch shape: {theta_batch.shape}")
print(f"   First 3 samples:")
print(theta_batch[:3])

# 3. Integration with scipy.optimize
print("\n3. Optimization with scipy:")


def objective_function(theta):
    """Wrapper for scipy.optimize"""
    params = bank.theta_to_instance(theta)
    return params["objective"]


# Initial guess
x0 = np.array([0.0, 0.0])

print(f"   Initial guess: x={x0[0]:.3f}, y={x0[1]:.3f}")
print(f"   Initial objective: {objective_function(x0):.3f}")

# Optimize
result = minimize(
    objective_function,
    x0,
    method="Nelder-Mead",
    bounds=[(-5.0, 5.0), (-5.0, 5.0)],
)

print(f"\n   Optimization result:")
print(f"     Success: {result.success}")
print(f"     Optimal x: {result.x[0]:.3f}")
print(f"     Optimal y: {result.x[1]:.3f}")
print(f"     Minimum value: {result.fun:.6f}")
print(f"     Expected: x=2.0, y=-1.0, f=0.0")

# 4. Working with bounds
print("\n4. Parameter bounds:")
print(f"   Lower bounds: {bank.lower_bounds}")
print(f"   Upper bounds: {bank.upper_bounds}")

# 5. Random sampling for Monte Carlo
print("\n5. Monte Carlo sampling:")
n_samples = 10000
samples = bank.sample(size=n_samples)
print(f"   Generated {n_samples} samples")
print(f"   Sample shape: {samples.shape}")

# Convert to full parameter sets and compute statistics
full_samples = [bank.theta_to_instance(theta) for theta in samples]
objectives = np.array([ps["objective"] for ps in full_samples])

print(f"\n   Objective function statistics:")
print(f"     Mean: {objectives.mean():.3f}")
print(f"     Std: {objectives.std():.3f}")
print(f"     Min: {objectives.min():.3f}")
print(f"     Max: {objectives.max():.3f}")

# Find best sample
best_idx = objectives.argmin()
best_theta = samples[best_idx]
best_params = bank.theta_to_instance(best_theta)
print(f"\n   Best sample from Monte Carlo:")
print(f"     x={best_params['x']:.3f}, y={best_params['y']:.3f}")
print(f"     objective={best_params['objective']:.3f}")

# 6. Multi-dimensional sampling for grid search
print("\n6. Multi-dimensional sampling:")
grid_samples = bank.sample(size=(10, 10))
print(f"   Grid shape: {grid_samples.shape}")
print(f"   This creates a 10x10 grid with 2 parameters each")

print("\n" + "=" * 60)
