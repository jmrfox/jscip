"""Example demonstrating IndependentVectorParameter usage in jscip.

This example shows how to use IndependentVectorParameter for array-valued parameters
with multivariate distributions.
"""

import numpy as np

from jscip import IndependentVectorParameter

print("=" * 70)
print("IndependentVectorParameter Examples")
print("=" * 70)

# Example 1: Basic vector parameter
print("\n1. Basic Vector Parameter")
print("-" * 70)

position = IndependentVectorParameter(
    value=[1.0, 2.0, 3.0],
    is_sampled=False,
)

print(f"Position: {position.value}")
print(f"Shape: {position.shape}")
print(f"Is sampled: {position.is_sampled}")

# Example 2: Uniform sampling with uniform range
print("\n2. Uniform Sampling with Uniform Range")
print("-" * 70)

velocity = IndependentVectorParameter(
    value=[0.0, 0.0, 0.0],
    is_sampled=True,
    range=(-10.0, 10.0),  # Same range for all components
    distribution="uniform",
)

print(f"Initial velocity: {velocity.value}")
print("Range (all components): [-10.0, 10.0]")

# Sample single vector
sample = velocity.sample()
print(f"Single sample: {sample}")

# Sample multiple vectors
samples = velocity.sample(size=5)
print(f"Multiple samples shape: {samples.shape}")
print(f"Multiple samples:\n{samples}")

# Example 3: Uniform sampling with element-wise ranges
print("\n3. Uniform Sampling with Element-wise Ranges")
print("-" * 70)

position_3d = IndependentVectorParameter(
    value=[0.0, 0.0, 0.0],
    is_sampled=True,
    range=(
        [-5.0, -3.0, -1.0],  # Lower bounds for x, y, z
        [5.0, 3.0, 1.0],  # Upper bounds for x, y, z
    ),
    distribution="uniform",
)

print("Element-wise ranges:")
print(f"  x: [{position_3d.range[0][0]}, {position_3d.range[1][0]}]")
print(f"  y: [{position_3d.range[0][1]}, {position_3d.range[1][1]}]")
print(f"  z: [{position_3d.range[0][2]}, {position_3d.range[1][2]}]")

samples = position_3d.sample(size=10)
print(f"\n10 samples:\n{samples}")

# Example 4: Multivariate normal distribution
print("\n4. Multivariate Normal Distribution")
print("-" * 70)

# Create a 2D parameter with correlation
correlated_params = IndependentVectorParameter(
    value=[0.0, 0.0],
    is_sampled=True,
    range=(-3.0, 3.0),
    distribution="mvnormal",
)

print("Distribution: Multivariate normal (independent components)")
print(f"Mean (center of range): {(correlated_params.range[0] + correlated_params.range[1]) / 2.0}")

samples = correlated_params.sample(size=100)
print(f"\n100 samples shape: {samples.shape}")
print(f"Sample mean: {samples.mean(axis=0)}")
print(f"Sample std: {samples.std(axis=0)}")

# Example 5: Multivariate normal with custom covariance
print("\n5. Multivariate Normal with Custom Covariance")
print("-" * 70)

# Define covariance matrix with correlation
cov = np.array(
    [
        [1.0, 0.8],  # High positive correlation
        [0.8, 1.0],
    ]
)

correlated = IndependentVectorParameter(
    value=[0.0, 0.0],
    is_sampled=True,
    range=(-5.0, 5.0),
    distribution="mvnormal",
    cov=cov,
)

print(f"Covariance matrix:\n{cov}")

samples = correlated.sample(size=1000)
print("\n1000 samples:")
print(f"  Sample correlation: {np.corrcoef(samples.T)[0, 1]:.3f}")
print(f"  Expected correlation: {cov[0, 1]:.3f}")

# Example 6: Updating vector parameter values
print("\n6. Updating Vector Parameter Values")
print("-" * 70)

param = IndependentVectorParameter(
    value=[1.0, 2.0, 3.0],
    range=(0.0, 5.0),
)

print(f"Initial value: {param.value}")

# Update with list
param.value = [2.0, 3.0, 4.0]
print(f"After update (list): {param.value}")

# Update with numpy array
param.value = np.array([1.5, 2.5, 3.5])
print(f"After update (array): {param.value}")

# Example 7: Copying vector parameters
print("\n7. Copying Vector Parameters")
print("-" * 70)

original = IndependentVectorParameter(
    value=[1.0, 2.0],
    is_sampled=True,
    range=(0.0, 5.0),
    distribution="mvnormal",
    cov=np.array([[1.0, 0.5], [0.5, 1.0]]),
)

copy = original.copy()

print(f"Original value: {original.value}")
print(f"Copy value: {copy.value}")

# Modify copy
copy.value = [3.0, 4.0]
print("\nAfter modifying copy:")
print(f"  Original value: {original.value}")
print(f"  Copy value: {copy.value}")

# Example 8: Physical simulation with vector parameters
print("\n8. Physical Simulation Example")
print("-" * 70)

# Initial position (fixed)
initial_position = IndependentVectorParameter(
    value=[0.0, 0.0, 10.0],  # Start at height 10m
    is_sampled=False,
)

# Initial velocity (sampled)
initial_velocity = IndependentVectorParameter(
    value=[5.0, 5.0, 0.0],
    is_sampled=True,
    range=(
        [0.0, 0.0, -2.0],  # vx, vy, vz ranges
        [10.0, 10.0, 2.0],
    ),
    distribution="uniform",
)

print("Projectile motion simulation:")
print(f"  Initial position: {initial_position.value}")
print("  Initial velocity range:")
print("    vx: [0.0, 10.0] m/s")
print("    vy: [0.0, 10.0] m/s")
print("    vz: [-2.0, 2.0] m/s")

# Sample 5 different initial velocities
velocities = initial_velocity.sample(size=5)
print(f"\n5 sampled initial velocities:\n{velocities}")

print("\n" + "=" * 70)
print("Examples complete!")
print("=" * 70)
