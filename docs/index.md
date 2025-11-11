# jscip

A lightweight framework for managing scientific/numerical parameters with support for sampling, derived parameters, constraints, and convenient conversions.

## Installation

- From source (this repo):

```bash
pip install -e .
```

This project uses Hatchling as the build backend (configured in `pyproject.toml`).

## Quickstart

```python
from jscip import IndependentParameter, DerivedParameter, ParameterBank

# Define parameters
p1 = IndependentParameter(value=0.5, is_sampled=True, range=(0.0, 1.0))
p2 = IndependentParameter(value=2.0)  # fixed input
prod = DerivedParameter(lambda ps: ps["p1"] * ps["p2"])  # derived

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

## Testing

```bash
pytest -q
```
