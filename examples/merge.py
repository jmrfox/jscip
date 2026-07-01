"""ParameterBank merge example for jscip.

This example demonstrates:
- Merging modular parameter banks
- Default collision behavior (error on duplicate names)
- Underwrite mode (keep existing, add new parameters)
- Overwrite mode (replace existing with incoming definitions)
- Composing banks into a full model
"""

from jscip import (
    DerivedScalarParameter,
    IndependentScalarParameter,
    ParameterBank,
)

# Build two modular banks that share a parameter name ("mass")
physics_bank = ParameterBank(
    parameters={
        "mass": IndependentScalarParameter(
            value=1.0, is_sampled=True, range=(0.5, 2.0)
        ),
        "gravity": IndependentScalarParameter(value=9.81, is_sampled=False),
    }
)

motion_bank = ParameterBank(
    parameters={
        "mass": IndependentScalarParameter(
            value=5.0, is_sampled=True, range=(4.0, 6.0)
        ),
        "velocity": IndependentScalarParameter(
            value=10.0, is_sampled=True, range=(5.0, 15.0)
        ),
    }
)

print("=" * 60)
print("ParameterBank Merge Example")
print("=" * 60)

# 1. Default behavior: collision raises KeyError
print("\n1. Default merge (on_collision='error'):")
receiver = physics_bank.copy()
incoming = motion_bank.copy()
print(f"   Receiver mass default: {receiver['mass'].value}")
print(f"   Incoming mass default: {incoming['mass'].value}")
try:
    receiver.merge(incoming)
except KeyError as exc:
    print(f"   Merge failed as expected: {exc}")

# 2. Underwrite: keep receiver values, add new parameters
print("\n2. Underwrite merge (on_collision='underwrite'):")
receiver = physics_bank.copy()
incoming = motion_bank.copy()
receiver.merge(incoming, on_collision="underwrite")
print(f"   mass kept from receiver: {receiver['mass'].value}")
print(f"   velocity added from incoming: {receiver['velocity'].value}")
print(f"   gravity unchanged: {receiver['gravity'].value}")
print(f"   Combined parameters: {receiver.names}")

# 3. Overwrite: incoming definitions replace existing ones
print("\n3. Overwrite merge (on_collision='overwrite'):")
receiver = physics_bank.copy()
incoming = motion_bank.copy()
print(f"   Before merge, mass default: {receiver['mass'].value}")
receiver.merge(incoming, on_collision="overwrite")
print(f"   After merge, mass default: {receiver['mass'].value}")
print(f"   velocity added: {receiver['velocity'].value}")

# 4. Compose disjoint banks into a full model
print("\n4. Composing a full model from disjoint banks:")
model = ParameterBank(
    parameters={
        "mass": IndependentScalarParameter(
            value=1.0, is_sampled=True, range=(0.5, 2.0)
        ),
    }
)
kinematics = ParameterBank(
    parameters={
        "velocity": IndependentScalarParameter(
            value=10.0, is_sampled=True, range=(5.0, 15.0)
        ),
    }
)
environment = ParameterBank(
    parameters={
        "gravity": IndependentScalarParameter(value=9.81, is_sampled=False),
    }
)

model.merge(kinematics)
model.merge(environment)

def kinetic_energy(ps):
    return 0.5 * ps["mass"] * ps["velocity"] ** 2


model.add_parameter("ke", DerivedScalarParameter(kinetic_energy))

sample = model.sample()
print(f"   Sampled parameters: {list(sample.keys())}")
print(f"   mass={sample['mass']:.3f}, velocity={sample['velocity']:.3f}")
print(f"   kinetic energy={sample['ke']:.3f}")

print("\n" + "=" * 60)
