# jscip Examples

This directory contains practical examples demonstrating various features of jscip.

## Running the Examples

All examples can be run directly with Python:

```bash
uv run python examples/basic_usage.py
```

Or with standard Python:

```bash
python examples/basic_usage.py
```

## Available Examples

### 1. `basic_usage.py`
**Demonstrates**: Core functionality and basic parameter operations

- Creating independent and derived parameters
- Building a parameter bank
- Sampling single and multiple configurations
- Working with different output formats (ParameterSet, DataFrame)
- Accessing parameter properties

**Good for**: First-time users learning the basics

### 2. `constraints.py`
**Demonstrates**: Working with parameter constraints

- Defining constraints on parameter combinations
- Sampling with automatic constraint satisfaction
- Handling constraint failures
- Validating parameter sets manually
- Computing log probabilities

**Good for**: Users who need to ensure parameter validity

### 3. `derived_parameters.py`
**Demonstrates**: Complex derived parameter computations

- Creating multiple derived parameters
- Chaining derived computations
- Using derived parameters in constraints
- Computing statistics on derived quantities
- Real-world physics example (projectile motion)

**Good for**: Users with complex parameter dependencies

### 4. `integration.py`
**Demonstrates**: Integration with scientific libraries

- Using `array_mode` mode for optimization
- Converting between representations (ParameterSet ↔ array ↔ DataFrame)
- Integration with `scipy.optimize`
- Working with NumPy arrays
- Monte Carlo sampling
- Multi-dimensional sampling for grid search

**Good for**: Users integrating jscip with optimization or MCMC libraries

## Example Output

Each example includes formatted output showing:
- Parameter values and configurations
- Sampling results
- Statistical summaries
- Validation results

## Tips

- Start with `basic_usage.py` to understand core concepts
- Review `constraints.py` if you need parameter validation
- Check `derived_parameters.py` for complex dependencies
- See `integration.py` for advanced use cases with scientific libraries

## Further Reading

- [Main Documentation](https://jmrfox.github.io/jscip/)
- [API Reference](https://jmrfox.github.io/jscip/api/)
- [GitHub Repository](https://github.com/jmrfox/jscip)
