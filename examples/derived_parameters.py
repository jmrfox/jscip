"""Derived parameters example for jscip.

This example demonstrates:
- Creating complex derived parameters
- Chaining derived computations
- Using derived parameters in constraints
- Computing statistics on derived quantities
"""

import numpy as np

from jscip import DerivedScalarParameter, IndependentScalarParameter, ParameterBank

# Define base parameters for a projectile motion problem
initial_velocity = IndependentScalarParameter(value=20.0, is_sampled=True, range=(10.0, 30.0))
launch_angle = IndependentScalarParameter(value=45.0, is_sampled=True, range=(30.0, 60.0))
gravity = IndependentScalarParameter(value=9.81, is_sampled=False)


# Define derived parameters
def compute_vx(params):
    """Horizontal velocity component"""
    angle_rad = np.deg2rad(params["launch_angle"])
    return params["initial_velocity"] * np.cos(angle_rad)


def compute_vy(params):
    """Vertical velocity component"""
    angle_rad = np.deg2rad(params["launch_angle"])
    return params["initial_velocity"] * np.sin(angle_rad)


def compute_flight_time(params):
    """Total time in air"""
    angle_rad = np.deg2rad(params["launch_angle"])
    vy = params["initial_velocity"] * np.sin(angle_rad)
    return 2 * vy / params["gravity"]


def compute_max_height(params):
    """Maximum height reached"""
    angle_rad = np.deg2rad(params["launch_angle"])
    vy = params["initial_velocity"] * np.sin(angle_rad)
    return (vy**2) / (2 * params["gravity"])


def compute_range(params):
    """Horizontal distance traveled"""
    angle_rad = np.deg2rad(params["launch_angle"])
    v0_squared = params["initial_velocity"] ** 2
    return v0_squared * np.sin(2 * angle_rad) / params["gravity"]


# Create derived parameter objects
vx = DerivedScalarParameter(compute_vx)
vy = DerivedScalarParameter(compute_vy)
flight_time = DerivedScalarParameter(compute_flight_time)
max_height = DerivedScalarParameter(compute_max_height)
projectile_range = DerivedScalarParameter(compute_range)

# Create parameter bank
bank = ParameterBank(
    parameters={
        "initial_velocity": initial_velocity,
        "launch_angle": launch_angle,
        "gravity": gravity,
        "vx": vx,
        "vy": vy,
        "flight_time": flight_time,
        "max_height": max_height,
        "range": projectile_range,
    },
    constraints=[
        lambda ps: ps["max_height"] < 50.0,  # Height limit
        lambda ps: ps["range"] > 10.0,  # Minimum range
    ],
)

print("=" * 60)
print("Derived Parameters Example: Projectile Motion")
print("=" * 60)

# Sample configurations
print("\n1. Single trajectory:")
trajectory = bank.sample()
print(f"   Initial velocity: {trajectory['initial_velocity']:.2f} m/s")
print(f"   Launch angle: {trajectory['launch_angle']:.1f}°")
print(f"   Horizontal velocity: {trajectory['vx']:.2f} m/s")
print(f"   Vertical velocity: {trajectory['vy']:.2f} m/s")
print(f"   Flight time: {trajectory['flight_time']:.2f} s")
print(f"   Max height: {trajectory['max_height']:.2f} m")
print(f"   Range: {trajectory['range']:.2f} m")

# Sample multiple and analyze
print("\n2. Statistical analysis of 1000 trajectories:")
trajectories = bank.sample(size=1000)

print("   Range statistics:")
print(f"     Mean: {trajectories['range'].mean():.2f} m")
print(f"     Std: {trajectories['range'].std():.2f} m")
print(f"     Min: {trajectories['range'].min():.2f} m")
print(f"     Max: {trajectories['range'].max():.2f} m")

print("\n   Max height statistics:")
print(f"     Mean: {trajectories['max_height'].mean():.2f} m")
print(f"     Std: {trajectories['max_height'].std():.2f} m")
print(f"     Min: {trajectories['max_height'].min():.2f} m")
print(f"     Max: {trajectories['max_height'].max():.2f} m")

# Find optimal angle for maximum range
print("\n3. Finding optimal launch angle:")
best_idx = trajectories["range"].idxmax()
best_trajectory = trajectories.loc[best_idx]
print(f"   Best angle: {best_trajectory['launch_angle']:.1f}°")
print(f"   Best range: {best_trajectory['range']:.2f} m")
print("   (Theoretical optimum is 45° for maximum range)")

# Demonstrate that derived parameters update correctly
print("\n4. Derived parameters update with base parameters:")
test_params = bank.get_default_values()
print(f"   Default angle: {test_params['launch_angle']:.1f}°")
print(f"   Default range: {test_params['range']:.2f} m")

print("\n" + "=" * 60)
