# jscip

A lightweight framework for managing scientific/numerical parameters with support for sampling, derived parameters, constraints, and convenient conversions.

## Installation

From source:

```bash
git clone https://github.com/jmrfox/jscip.git
cd jscip
pip install .
```

Or using `uv`:

```bash
git clone https://github.com/jmrfox/jscip.git
cd jscip
uv pip install .
```

For development (editable install):

```bash
pip install -e .
# or: uv pip install -e .
```

This project uses Hatchling as the build backend (configured in `pyproject.toml`).

## Quickstart

### Scalar Parameters

```python
from jscip import IndependentScalarParameter, DerivedScalarParameter, ParameterBank

# Define parameters
p1 = IndependentScalarParameter(value=0.5, is_sampled=True, range=(0.0, 1.0))
p2 = IndependentScalarParameter(value=2.0)  # fixed input
prod = DerivedScalarParameter(lambda ps: ps["p1"] * ps["p2"])  # derived

# Build a bank
bank = ParameterBank(
    parameters={"p1": p1, "p2": p2, "prod": prod},
    constraints=[lambda ps: ps["p1"] >= 0.2],
)

# Sample
sample = bank.sample()            # ParameterSet
batch = bank.sample(size=100)     # pandas.DataFrame
print(sample)
print(batch.head())
```

### Vector Parameters

```python
import numpy as np
from jscip import (
    IndependentVectorParameter,
    DerivedVectorParameter,
    ParameterBank,
)

# Define vector parameters
position = IndependentVectorParameter(
    value=[1.0, 2.0, 3.0],
    is_sampled=True,
    range=(0.0, 10.0),  # uniform range for all elements
    distribution="uniform"
)

# Derived vector parameter
def compute_velocity(ps):
    return np.array([ps["vx"], ps["vy"], ps["vz"]])

velocity = DerivedVectorParameter(
    function=compute_velocity,
    output_shape=(3,)
)

# Vector constraints
def norm_constraint(ps):
    return np.linalg.norm(ps["position"]) < 15.0

bank = ParameterBank(
    parameters={
        "position": position,
        "vx": IndependentScalarParameter(1.0, is_sampled=True, range=(-5.0, 5.0)),
        "vy": IndependentScalarParameter(0.0, is_sampled=True, range=(-5.0, 5.0)),
        "vz": IndependentScalarParameter(0.0, is_sampled=True, range=(-5.0, 5.0)),
        "velocity": velocity,
    },
    constraints=[norm_constraint]
)

sample = bank.sample()
print(f"Position: {sample['position']}")
print(f"Velocity: {sample['velocity']}")
```

## Testing

```bash
pytest -q
```
