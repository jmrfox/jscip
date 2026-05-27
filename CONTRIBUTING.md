# Contributing to JSCIP

Thank you for your interest in contributing to JSCIP! This document provides guidelines for contributors to ensure a smooth development process.

## Development Setup

### Prerequisites

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Setting Up the Development Environment

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jmrfox/jscip.git
   cd jscip
   ```

2. **Create a virtual environment:**
   ```bash
   # Using uv (recommended)
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   
   # Or using venv
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   # Using uv (recommended)
   uv pip install -e ".[dev]"
   
   # Or using pip
   pip install -e ".[dev]"
   ```

### Development Tools

JSCIP uses the following tools for code quality:

- **ruff**: Linting and formatting
- **mypy**: Type checking
- **pytest**: Testing

## Code Style and Quality

### Formatting and Linting

We use ruff for code formatting and linting. Before submitting a pull request:

```bash
# Check for issues
uv run ruff check

# Format code
uv run ruff format
```

### Type Checking

We use mypy for static type checking:

```bash
uv run mypy jscip/
```

### Testing

Run the test suite before submitting:

```bash
# Run all tests
uv run test

# Run specific test file
uv run pytest test/test_parameter_bank.py

# Run with coverage
uv run pytest --cov=jscip
```

## Contribution Guidelines

### Making Changes

1. **Create a new branch** for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the existing code style and patterns.

3. **Add tests** for new functionality. We aim for high test coverage.

4. **Update documentation** if your changes affect the API.

5. **Run the full test suite** and ensure all tests pass.

6. **Check code quality**:
   ```bash
   uv run ruff check
   uv run mypy jscip/
   ```

### Code Standards

- **Type hints**: All public APIs must have complete type annotations
- **Docstrings**: Use Google-style docstrings for all public methods and classes
- **Line length**: Maximum 100 characters (configured in ruff)
- **Imports**: Use `from __future__ import annotations` and organize imports properly

### Testing Standards

- **Unit tests**: Test individual components in isolation
- **Integration tests**: Test complete workflows
- **Edge cases**: Include tests for error conditions and edge cases
- **Documentation**: Complex tests should have explanatory comments

## Submitting Changes

### Pull Request Process

1. **Update your branch** with the latest changes:
   ```bash
   git fetch origin
   git rebase origin/main
   ```

2. **Create a pull request** with:
   - Clear title and description
   - Reference any relevant issues
   - Include screenshots if applicable

3. **Ensure CI checks pass** on your pull request.

4. **Address any feedback** from maintainers.

### Code Review Guidelines

- **Be constructive**: Provide helpful, specific feedback
- **Be thorough**: Check for bugs, performance issues, and code quality
- **Be respectful**: Maintain a positive and collaborative tone

## Project Structure

```
jscip/
├── jscip/                 # Main package
│   ├── __init__.py       # Package exports
│   ├── parameters.py     # Parameter classes
│   ├── parameter_bank.py # ParameterBank class
│   ├── parameter_set.py  # ParameterSet class
│   └── hypergrid.py      # HyperGrid class
├── test/                 # Test suite
├── examples/             # Usage examples
├── docs/                 # Documentation
└── pyproject.toml       # Project configuration
```

## Development Guidelines

### Adding New Features

1. **Design first**: Consider the API design and how it fits with existing code
2. **Document**: Add comprehensive docstrings and examples
3. **Test**: Write tests before or alongside implementation
4. **Type safety**: Ensure proper type annotations throughout

### Bug Fixes

1. **Reproduce**: Create a test that reproduces the issue
2. **Fix**: Implement the minimal fix for the issue
3. **Verify**: Ensure the fix doesn't break existing functionality
4. **Document**: Add comments explaining the fix if necessary

### Performance

- **Profile**: Use profiling tools to identify bottlenecks
- **Benchmark**: Add performance tests for critical paths
- **Optimize**: Focus on algorithmic improvements over micro-optimizations

## Getting Help

- **Issues**: Use GitHub issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Documentation**: Check the existing documentation and examples

## License

By contributing to JSCIP, you agree that your contributions will be licensed under the MIT License, as specified in the [LICENSE](LICENSE) file.

Thank you for contributing to JSCIP!
