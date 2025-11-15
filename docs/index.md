# jscip

A lightweight framework for managing scientific/numerical parameters with support for sampling, derived parameters, constraints, and convenient conversions between rich objects and arrays/data frames.

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

# Human-readable summary (pretty_print has been removed)
print(bank.summary())

# Sample
sample = bank.sample()            # ParameterSet
batch = bank.sample(size=100)     # pandas.DataFrame
print(sample)
print(batch.head())
```

### Parameter types and distributions

`IndependentParameter` represents a single scalar parameter. It supports both
floating-point and integer types:

- If you do not specify `param_type`, the type is inferred from `value`:
  - `int` value → integer parameter
  - `float` value → floating parameter
- You can also pass `param_type=float` or `param_type=int` explicitly.

When `is_sampled=True`, a SciPy distribution is constructed under the hood:

- **Continuous (float) parameters** support:
  - Uniform over a range (via `range` or `{"kind": "uniform", "low": ..., "high": ...}`)
  - Normal: `{"kind": "normal", "loc": ..., "scale": ...}`
  - Log-normal: `{"kind": "lognormal", "s": ..., "scale": ...}`
  - Exponential: `{"kind": "exponential", "scale": ...}`
  - Gamma: `{"kind": "gamma", "a": ..., "scale": ...}`
- **Discrete (integer) parameters** support:
  - Uniform over a range (via `range`)
  - Bernoulli: `{"kind": "bernoulli", "p": ...}`
  - Poisson: `{"kind": "poisson", "mu": ...}`
  - Binomial: `{"kind": "binomial", "n": ..., "p": ...}`
  - Discrete uniform: `{"kind": "discrete_uniform", "low": ..., "high": ...}`
  - Geometric: `{"kind": "geometric", "p": ...}`

You can also pass a user-supplied *frozen* SciPy distribution object as
`distribution`, in which case that object is used directly. Incompatible
combinations of `param_type` and distribution kind (for example, an
integer parameter with a normal distribution config) raise `ValueError`.

Sampling respects the parameter type: integer parameters return integer arrays,
and floats return floating arrays.

### Units

`IndependentParameter` accepts an optional `unit` object (for example from
`units-llnl`). Units are not used in internal computations, but can be applied
on readout:

- `param.sample()` returns plain numeric values.
- `param.sample(return_unit=True)` multiplies by the unit object, relying on
  its multiplication semantics.

At the `ParameterBank` level:

- `bank.get_default_values()` returns a numeric `ParameterSet`.
- `bank.get_default_values(with_units=True)` returns a `ParameterSet` where
  independent parameters with units are converted to unitful quantities.
- `bank.sample(with_units=True)` (with `size=None`) returns a single
  `ParameterSet` with unitful independent parameters; batch sampling still
  returns numeric `pandas.DataFrame`s.

### Vector mode and log-probabilities

`ParameterBank` supports a *vector mode* where NumPy arrays contain only sampled
independent parameters in a canonical order. This is useful for optimizers or
MCMC samplers that operate on numeric vectors:

- Set `bank.vector_mode = True`.
- `bank.sample(size=N)` then returns an `N × n_sampled` array.
- `bank.vector_to_instance(vector)` converts a vector back to a full
  `ParameterSet`, recomputing derived parameters.

The `log_prob` method provides a simple uniform prior over the bounds of sampled
independent parameters and any constraints:

- Returns `0.0` for instances that lie within all ranges and satisfy all
  constraints.
- Returns `-np.inf` otherwise.

## Testing

```bash
pytest -q
```
