# Jordan's Scientific Parameters

```text
                               /$$          
                              |__/          
       /$$  /$$$$$$$  /$$$$$$$ /$$  /$$$$$$ 
      |__/ /$$_____/ /$$_____/| $$ /$$__  $$
       /$$|  $$$$$$ | $$      | $$| $$  \ $$
      | $$ \____  $$| $$      | $$| $$  | $$
      | $$ /$$$$$$$/|  $$$$$$$| $$| $$$$$$$/
      | $$|_______/  \_______/|__/| $$____/
 /$$  | $$                        | $$
|  $$$$$$/                        | $$
 \______/                         |__/
```

[Documentation](https://jmrfox.github.io/jscip/)

A lightweight Python framework for managing numerical parameters in scientific workflows. Define parameters with ranges, constraints, and dependencies, then sample configurations for simulations, optimization, or uncertainty quantification.

## Features

- **Independent Parameters**: Real-valued parameters with optional uniform sampling over ranges
- **Derived Parameters**: Computed parameters that depend on other parameters
- **Constraints**: Define validity conditions for parameter combinations
- **Flexible Sampling**: Sample single instances or batches, with automatic constraint satisfaction
- **Multiple Representations**: Convert between rich objects, NumPy arrays, and pandas DataFrames
- **Type-Safe**: Full type hints for Python 3.12+

## Installation

### From source

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

### For development (editable install)

```bash
git clone https://github.com/jmrfox/jscip.git
cd jscip
pip install -e .
# or: uv pip install -e .
```

## Quick Start

```python
from jscip import IndependentScalarParameter, DerivedScalarParameter, ParameterBank

# Define independent parameters
mass = IndependentScalarParameter(value=1.0, is_sampled=True, range=(0.5, 2.0))
velocity = IndependentScalarParameter(value=10.0, is_sampled=True, range=(5.0, 15.0))
time = IndependentScalarParameter(value=1.0, is_sampled=False)

# Define a derived parameter
def compute_kinetic_energy(params):
    return 0.5 * params["mass"] * params["velocity"] ** 2

kinetic_energy = DerivedScalarParameter(compute_kinetic_energy)

# Create a parameter bank
bank = ParameterBank(
    parameters={
        "mass": mass,
        "velocity": velocity,
        "time": time,
        "kinetic_energy": kinetic_energy,
    },
    constraints=[
        lambda ps: ps["kinetic_energy"] < 100.0,  # Energy constraint
    ],
)

# Sample a single parameter set
sample = bank.sample()
print(sample)
# ParameterSet(mass=1.23, velocity=8.45, time=1.0, kinetic_energy=43.91)

# Sample multiple configurations as a DataFrame
samples_df = bank.sample(size=100)
print(samples_df.head())
```

## Basic Usage

### Creating Parameters

```python
from jscip import IndependentScalarParameter, DerivedScalarParameter

# Fixed parameter (not sampled)
fixed_param = IndependentScalarParameter(value=5.0)

# Sampled parameter with range
sampled_param = IndependentScalarParameter(
    value=1.0, 
    is_sampled=True, 
    range=(0.0, 2.0)
)

# Derived parameter
derived = DerivedScalarParameter(lambda ps: ps["param1"] + ps["param2"])
```

### Working with Parameter Banks

```python
from jscip import ParameterBank

# Create a bank
bank = ParameterBank(
    parameters={"p1": param1, "p2": param2},
    constraints=[lambda ps: ps["p1"] < ps["p2"]],
    max_attempts=100,  # Max attempts to satisfy constraints
)

# Add parameters dynamically
bank.add_parameter("p3", param3)
bank.add_constraint(lambda ps: ps["p3"] > 0)

# Sample with different modes
single = bank.sample()                    # Single ParameterSet
batch = bank.sample(size=50)              # DataFrame with 50 samples
array = bank.sample(size=(10, 5))         # 3D array (requires theta_sampling=True)
```

### Conversions

```python
# Convert between representations
theta_array = bank.instance_to_theta(sample)      # ParameterSet → array
sample_back = bank.theta_to_instance(theta_array) # array → ParameterSet
theta_from_df = bank.dataframe_to_theta(df)       # DataFrame → array

# Get default values
defaults = bank.get_default_values()
```

### Constraints and Validation

```python
# Check if a sample satisfies constraints
is_valid = sample.satisfies(lambda ps: ps["x"] > 0)

# Compute log probability (uniform prior)
log_prob = bank.log_prob(sample)  # 0.0 if valid, -inf if invalid
```

## Advanced Features

### Theta Sampling Mode

For integration with optimization or MCMC libraries that expect flat arrays:

```python
bank = ParameterBank(
    parameters={"a": param_a, "b": param_b},
    theta_sampling=True,  # Only sample/return sampled parameters
)

theta = bank.sample()  # Returns 1D array instead of ParameterSet
```

### TeX Names for Plotting

```python
bank = ParameterBank(
    parameters={"alpha": param1, "beta": param2},
    texnames={"alpha": r"$\alpha$", "beta": r"$\beta$"},
)

# Use for plot labels
labels = bank.sampled_texnames
```

## Examples

See the [`examples/`](examples/) directory for complete workflows:

- `basic_usage.py` - Parameter definition and sampling
- `constraints.py` - Working with constraints
- `derived_parameters.py` - Computing derived quantities
- `integration.py` - Integration with scientific libraries

## API Reference

Full API documentation is available at [https://jmrfox.github.io/jscip/](https://jmrfox.github.io/jscip/)

## Contributing

Contributions are welcome! Please see the [TODO.md](TODO.md) for planned features and improvements.

## License

This package is distributed under the MIT License.
