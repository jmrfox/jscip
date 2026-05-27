"""HyperGrid usage example for jscip.

This example demonstrates:
- Creating a ParameterBank with sampled scalar parameters
- Setting up a HyperGrid for systematic evaluation
- Working with derived parameters in the grid
- Memory considerations for large grids
"""

from jscip import (
    DerivedScalarParameter,
    HyperGrid,
    IndependentScalarParameter,
    ParameterBank,
)

print("=" * 60)
print("HyperGrid Example: Systematic Parameter Evaluation")
print("=" * 60)

# Define independent scalar parameters
mass = IndependentScalarParameter(value=1.0, is_sampled=True, range=(0.5, 2.0))
velocity = IndependentScalarParameter(
    value=10.0, is_sampled=True, range=(5.0, 15.0)
)
gravity = IndependentScalarParameter(
    value=9.81, is_sampled=False
)  # Fixed parameter


# Define derived parameters
def compute_kinetic_energy(params):
    """Compute kinetic energy: KE = 0.5 * m * v^2"""
    return 0.5 * params["mass"] * params["velocity"] ** 2


def compute_potential_energy(params):
    """Compute potential energy: PE = m * g * h (assuming h=1 unit)"""
    return params["mass"] * params["gravity"] * 1.0


def compute_total_energy(params):
    """Compute total mechanical energy."""
    return params["kinetic_energy"] + params["potential_energy"]


kinetic_energy = DerivedScalarParameter(compute_kinetic_energy)
potential_energy = DerivedScalarParameter(compute_potential_energy)
total_energy = DerivedScalarParameter(compute_total_energy)

# Create a parameter bank
bank = ParameterBank(
    parameters={
        "mass": mass,
        "velocity": velocity,
        "gravity": gravity,
        "kinetic_energy": kinetic_energy,
        "potential_energy": potential_energy,
        "total_energy": total_energy,
    }
)

print("\n1. Parameter Bank Configuration:")
print(f"   Sampled parameters: {bank.sampled}")
derived_params = [
    name
    for name, param in bank.parameters.items()
    if hasattr(param, "compute")
]
print(f"   Derived parameters: {derived_params}")
fixed_params = [
    name
    for name, param in bank.parameters.items()
    if (not param.is_sampled) and (not hasattr(param, "compute"))
]
print(f"   Fixed parameters: {fixed_params}")

# Create a HyperGrid with 3 points per parameter
print("\n2. Creating HyperGrid with 3 points per parameter:")
grid = HyperGrid(bank, n_points=3)
print(f"   Grid parameters: {list(grid.grid_params.keys())}")
print(f"   Total grid points: {3 ** len(grid.grid_params)}")

# Estimate memory usage
memory_mb = grid._estimate_memory_usage()
print(f"   Estimated memory usage: {memory_mb:.2f} MB")

# Generate the grid
print("\n3. Generating grid points:")
grid_points = grid.generate()
print(f"   Generated {len(grid_points)} grid points")

# Display first few grid points
print("\n4. First 5 grid points:")
print("   Mass  | Velocity | KE      | PE      | Total E")
print("   -------|----------|---------|---------|----------")
for i, point in enumerate(grid_points[:5]):
    format_str = (
        f"   {point['mass']:5.2f} | {point['velocity']:8.2f} | "
        f"{point['kinetic_energy']:7.2f} | {point['potential_energy']:7.2f} | "
        f"{point['total_energy']:8.2f}"
    )
    print(format_str)

# Find the point with maximum total energy
print("\n5. Analysis of grid results:")
max_energy_point = max(grid_points, key=lambda p: p["total_energy"])
min_energy_point = min(grid_points, key=lambda p: p["total_energy"])

print(f"   Maximum total energy: {max_energy_point['total_energy']:.2f}")
print(
    f"     at mass={max_energy_point['mass']:.2f}, velocity={max_energy_point['velocity']:.2f}"
)
print(f"   Minimum total energy: {min_energy_point['total_energy']:.2f}")
print(
    f"     at mass={min_energy_point['mass']:.2f}, velocity={min_energy_point['velocity']:.2f}"
)

# Example with larger grid (will show memory warning)
print("\n6. Example with larger grid (5 points per parameter):")
large_grid = HyperGrid(bank, n_points=5)
large_memory_mb = large_grid._estimate_memory_usage()
print(f"   Grid points: {5 ** len(large_grid.grid_params)}")
print(f"   Estimated memory: {large_memory_mb:.2f} MB")

if large_memory_mb > 100:
    print("   ⚠️  This grid is large and may consume significant memory!")
else:
    print("   Generating grid...")
    large_points = large_grid.generate()
    print(f"   Successfully generated {len(large_points)} points")

# Example of error handling
print("\n7. Error handling examples:")

# Try to create grid with vector parameter (should fail)
try:
    from jscip import IndependentVectorParameter

    vector_param = IndependentVectorParameter(
        value=[1.0, 2.0], is_sampled=True, range=(0.0, 3.0)
    )
    bad_bank = ParameterBank({"vector": vector_param})
    HyperGrid(bad_bank, n_points=3)
except ValueError as e:
    print(f"   ✅ Vector parameter correctly rejected: {e}")

# Try to create grid with no sampled parameters (should fail)
try:
    fixed_param1 = IndependentScalarParameter(value=1.0, is_sampled=False)
    fixed_param2 = IndependentScalarParameter(value=2.0, is_sampled=False)
    bad_bank2 = ParameterBank({"p1": fixed_param1, "p2": fixed_param2})
    HyperGrid(bad_bank2, n_points=3)
except ValueError as e:
    print(f"   ✅ No sampled parameters correctly rejected: {e}")

print("\n" + "=" * 60)
print("HyperGrid Example Complete")
print("=" * 60)
