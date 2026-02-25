"""Constraint handling example for jscip.

This example demonstrates:
- Defining constraints on parameter combinations
- Sampling with constraint satisfaction
- Handling constraint failures
- Validating parameter sets
"""

from jscip import DerivedScalarParameter, IndependentScalarParameter, ParameterBank

# Define parameters for a simple physics problem
# We want to ensure total energy stays within bounds
potential_energy = IndependentScalarParameter(
    value=50.0, is_sampled=True, range=(0.0, 100.0)
)
kinetic_energy = IndependentScalarParameter(
    value=30.0, is_sampled=True, range=(0.0, 100.0)
)


def compute_total_energy(params):
    """Total energy is sum of potential and kinetic"""
    return params["potential_energy"] + params["kinetic_energy"]


total_energy = DerivedScalarParameter(compute_total_energy)

print("=" * 60)
print("Constraint Handling Example")
print("=" * 60)

# Create bank with constraints
bank = ParameterBank(
    parameters={
        "potential_energy": potential_energy,
        "kinetic_energy": kinetic_energy,
        "total_energy": total_energy,
    },
    constraints=[
        # Total energy must be between 40 and 120
        lambda ps: ps["total_energy"] >= 40.0,
        lambda ps: ps["total_energy"] <= 120.0,
        # Kinetic energy must be at least 20% of total
        lambda ps: ps["kinetic_energy"] >= 0.2 * ps["total_energy"],
    ],
    max_attempts=100,
)

print("\n1. Sampling with constraints:")
print("   Constraints:")
print("   - Total energy between 40 and 120")
print("   - Kinetic energy >= 20% of total energy")

# Sample valid configurations
samples = bank.sample(size=10)
print(f"\n   Successfully sampled {len(samples)} valid configurations:")
print(samples[["potential_energy", "kinetic_energy", "total_energy"]].head())

# Verify constraints are satisfied
print("\n2. Verifying constraints:")
for idx, row in samples.head(3).iterrows():
    ps = row
    total = ps["total_energy"]
    ke_ratio = ps["kinetic_energy"] / total
    print(f"   Sample {idx}: Total={total:.1f}, KE ratio={ke_ratio:.2%}")

# Check log probability
print("\n3. Log probability (uniform prior):")
valid_sample = samples.iloc[0]
print(f"   Valid sample: {bank.log_prob(valid_sample):.1f}")

# Create an invalid sample (violates constraints)
invalid_sample = valid_sample.copy()
invalid_sample["kinetic_energy"] = 5.0  # Too low relative to total
invalid_sample["total_energy"] = 55.0
print(f"   Invalid sample: {bank.log_prob(invalid_sample):.1f}")

# Demonstrate constraint failure handling
print("\n4. Handling impossible constraints:")
impossible_bank = ParameterBank(
    parameters={
        "x": IndependentScalarParameter(0.5, is_sampled=True, range=(0.0, 1.0)),
    },
    constraints=[
        lambda ps: ps["x"] > 10.0,  # Impossible!
    ],
    max_attempts=5,
)

try:
    impossible_bank.sample()
except RuntimeError as e:
    print(f"   Expected error: {e}")

# Using satisfies method
print("\n5. Manual constraint checking:")
test_sample = bank.sample()


def custom_constraint(ps):
    return ps["potential_energy"] > ps["kinetic_energy"]


satisfies = test_sample.satisfies(custom_constraint)
print(f"   PE > KE? {satisfies}")
print(
    f"   PE={test_sample['potential_energy']:.1f}, KE={test_sample['kinetic_energy']:.1f}"
)

print("\n" + "=" * 60)
