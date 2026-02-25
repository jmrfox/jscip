"""Basic usage example for jscip.

This example demonstrates:
- Creating independent and derived parameters
- Building a parameter bank
- Sampling parameter configurations
- Working with different output formats
"""

from jscip import DerivedScalarParameter, IndependentScalarParameter, ParameterBank

# Define independent parameters
mass = IndependentScalarParameter(value=1.0, is_sampled=True, range=(0.5, 2.0))
velocity = IndependentScalarParameter(value=10.0, is_sampled=True, range=(5.0, 15.0))
time = IndependentScalarParameter(value=1.0, is_sampled=False)


# Define derived parameters
def compute_kinetic_energy(params):
    """Compute kinetic energy: KE = 0.5 * m * v^2"""
    return 0.5 * params["mass"] * params["velocity"] ** 2


def compute_distance(params):
    """Compute distance: d = v * t"""
    return params["velocity"] * params["time"]


kinetic_energy = DerivedScalarParameter(compute_kinetic_energy)
distance = DerivedScalarParameter(compute_distance)

# Create a parameter bank
bank = ParameterBank(
    parameters={
        "mass": mass,
        "velocity": velocity,
        "time": time,
        "kinetic_energy": kinetic_energy,
        "distance": distance,
    }
)

print("=" * 60)
print("Basic Parameter Sampling Example")
print("=" * 60)

# Sample a single parameter set
print("\n1. Single sample:")
sample = bank.sample()
print(sample)
print(f"   Kinetic Energy: {sample['kinetic_energy']:.2f}")
print(f"   Distance: {sample['distance']:.2f}")

# Sample multiple configurations as a DataFrame
print("\n2. Multiple samples (DataFrame):")
samples_df = bank.sample(size=5)
print(samples_df)

# Get default values
print("\n3. Default values:")
defaults = bank.get_default_values()
print(defaults)

# Access individual parameters
print("\n4. Parameter properties:")
print(f"   Mass is sampled: {bank['mass'].is_sampled}")
print(f"   Time is sampled: {bank['time'].is_sampled}")
print(f"   Sampled parameters: {bank.sampled}")
print(f"   All parameters: {bank.names}")

# Copy a parameter
print("\n5. Copying parameters:")
mass_copy = mass.copy()
print(f"   Original mass: {mass}")
print(f"   Copied mass: {mass_copy}")

print("\n" + "=" * 60)
