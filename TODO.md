# TODO List for jscip

## High Priority

### 1. Module Organization
- [x] Split `main.py` into logical modules:
  - [x] `parameters.py` - `IndependentScalarParameter`, `DerivedScalarParameter`
  - [x] `parameter_set.py` - `ParameterSet`
  - [x] `parameter_bank.py` - `ParameterBank`
  - [x] `vector_parameter.py` - `IndependentVectorParameter`
  - [ ] `constraints.py` - Constraint-related utilities (optional)
- [x] Update `__init__.py` to import from new modules
- [x] Rename parameter classes for clarity:
  - [x] `IndependentParameter` → `IndependentScalarParameter`
  - [x] `VectorParameter` → `IndependentVectorParameter`
  - [x] `DerivedParameter` → `DerivedScalarParameter`
  - [x] Update all imports and references in core modules
  - [x] Update all test files
  - [x] Update all example files
  - [x] Update README.md
  - [x] Update TODO.md
  - [x] Update docstrings and comments
  - [x] All 71 tests passing

### 2. Test Coverage Expansion
- [x] Add tests for `DerivedScalarParameter` class
- [x] Add tests for `ParameterBank.merge()`
- [x] Add tests for `ParameterBank.add_parameter()`
- [x] Add tests for `ParameterBank.add_constraint()`
- [x] Add tests for `array_to_instance()` conversions
- [x] Add tests for `dataframe_to_array()` conversions
- [x] Add edge case tests:
  - [x] Empty banks
  - [x] Constraint failures
  - [x] Multi-dimensional sampling
- [x] Add tests for `pretty_print()` method
- [x] Add integration tests for complete workflows

### 3. Bug Fixes
- [x] Fix constraint checking in `_log_prob_single()` (line 870-871)
  - Constraints are checked inside parameter loop, causing redundant checks
  - Move constraint checking outside the loop

### 4. Configuration & Tooling
- [x] Add `[tool.ruff]` configuration to `pyproject.toml`
  - Set line length, select rules, exclude patterns
- [x] Add `[tool.mypy]` configuration to `pyproject.toml`
  - Enable strict mode
  - Configure ignore patterns
- [x] Verify CI/CD pipeline runs tests, linting, and type checking
  - Added lint job with ruff check and format
  - Added type-check job with mypy
  - Enhanced test job with verbose output
- [x] Fix pytest configuration for `uv run` compatibility
  - Added `[tool.pytest.ini_options]` with testpaths
  - Added `[project.scripts]` entry for `uv run test` command

## Medium Priority

### 5. Code Quality Improvements
- [x] Remove commented debug code:
  - Line 716-717: `# print("n_samples (type):", n_samples, type(n_samples))`
  - Line 829: `# print("Converting 1D numpy array to ParameterSet instance.")`
  - Line 840: `# print("Converting 2D numpy array to list of ParameterSet instances.")`
- [x] Replace with proper logging where needed

### 6. Make `_max_attempts` Configurable
- [x] Add `max_attempts` parameter to `ParameterBank.__init__()`
- [x] Update docstring
- [x] Add validation
- [x] Add tests for custom max_attempts

### 7. Type Hints Improvements
- [x] Add complete type annotations to `ParameterSet.__init__()`
- [ ] Review and add missing return type hints throughout
- [ ] Ensure consistency with Python 3.12+ type syntax

### 8. Error Message Fixes
- [x] Fix "theta_array" → "return_array" in error message (line 481)
- [ ] Review all error messages for consistency and clarity

### 9. Documentation Expansion
- [x] Expand README.md with:
  - [x] Installation instructions (pip, uv)
  - [x] Quick start guide
  - [x] Basic usage examples
  - [x] API overview
  - [x] Link to full documentation
- [x] Create `examples/` directory with:
  - [x] Basic parameter definition
  - [x] Sampling with constraints
  - [x] Derived parameters usage
  - [x] Integration with scientific workflows

### 10. API Design Improvements
- [x] Add `@property` decorator for `is_sampled` on `IndependentScalarParameter`
- [x] Add `@property` decorator for `is_sampled` on `DerivedScalarParameter`
- [x] Update all direct `._is_sampled` accesses to use property
- [x] Add validation for `texnames` dictionary in `ParameterBank.__init__()`
  - Ensure keys match parameter names
  - Add helpful error messages

## Low Priority

### 11. Refactoring for Clarity
- [x] Simplify `isinstance` checks using tuple syntax
  - Example: `isinstance(value, (IndependentScalarParameter, DerivedScalarParameter))`
- [ ] Consider using `dataclasses` for cleaner initialization

### 12. Performance Optimizations
- [ ] Vectorize sampling loop for large samples without constraints
- [ ] Optimize repeated dictionary comprehensions in:
  - `_sample_once()`
  - `get_default_values()`
- [ ] Consider caching frequently accessed properties

## New Feature: Vector/Array Parameters

### 13. Implement Vector Parameter Support
Vector parameters will allow parameters to be Python iterables or NumPy arrays rather than just scalars.

#### Design Considerations
- [x] **Class Design Decision**:
  - ✓ Option A: New `IndependentVectorParameter` class parallel to `IndependentScalarParameter`
  - Option B: New `ArrayParameter` class with different semantics
  - Option C: Extend `IndependentScalarParameter` to support both scalar and vector modes
  - **Decision**: Option A implemented for clarity and type safety

#### Implementation Steps

##### Phase 1: Core IndependentVectorParameter Class ✅ COMPLETE
- [x] Create `IndependentVectorParameter` class with:
  - [x] `shape: tuple[int]` - Dimensionality (restricted to Nx1 vectors)
  - [x] `value: np.ndarray` - Current vector value (1D array)
  - [x] `is_sampled: bool` - Whether to sample this parameter
  - [x] `range: tuple[np.ndarray, np.ndarray] | tuple[float, float] | None` - Element-wise or uniform bounds
  - [x] `distribution: str` - Distribution type ('uniform' or 'mvnormal')
  - [x] Validation logic for shape consistency and range specifications
  - [x] `sample()` method with multivariate distribution support
  - [x] `copy()` method with covariance preservation
  - [x] 24 comprehensive tests - all passing

##### Phase 2: ParameterSet Integration ✅ COMPLETE
- [x] Extend `ParameterSet` to handle mixed scalar/vector parameters:
  - [x] Update storage to support both scalar and array values
  - [x] Ensure indexing works correctly (`ps["vector_param"]` returns array)
  - [x] Update `satisfies()` to work with vector-aware constraints
  - [x] Update `reindex()` to preserve array types
  - [x] Update `copy()` to deep copy numpy arrays
  - [x] Update docstring to reflect vector support
  - [x] 12 comprehensive tests - all passing

##### Phase 3: ParameterBank Integration ✅ COMPLETE
- [x] Update `ParameterBank` to support `IndependentVectorParameter`:
  - [x] Modify `__init__()` to accept `IndependentVectorParameter` instances
  - [x] Update `_refresh_sampled_indices()` to handle vector parameters
  - [x] Update `sampled` property to include vector parameters
  - [x] Add `vector_names` property to list vector parameter names
  - [x] Update `lower_bounds` and `upper_bounds` to handle vectors
    - Returns flattened array with all scalar and vector bounds concatenated
  - [x] Update `add_parameter()` to accept vector parameters
  - [x] Update `__getitem__()` return type annotation
  - [x] 11 comprehensive tests - all passing

##### Phase 4: Sampling & Conversion ✅ COMPLETE
- [x] Update sampling methods:
  - [x] Modify `_sample_once()` to sample vector parameters
  - [x] `_sample_once_constrained()` works with vectors (inherits from `_sample_once()`)
  - [x] Update `sample()` to return appropriate structures with correct theta dimensions
- [x] Update conversion methods:
  - [x] `instance_to_array()` - Flattens vectors into theta array
  - [x] `array_to_instance()` - Reshapes flat arrays back to vectors
  - [x] `dataframe_to_array()` - Works with existing implementation
  - [x] `instances_to_dataframe()` - Stores arrays as object dtype (Option A)
  - [x] `get_default_values()` - Updated to include vector parameters
- [x] 12 comprehensive tests - all passing

##### Phase 5: Derived Parameters with Vectors ✅ COMPLETE
- [x] Created base class hierarchy:
  - [x] `IndependentParameter` base class for scalar and vector independent parameters
  - [x] `DerivedParameter` base class for scalar and vector derived parameters
  - [x] Refactored existing classes to inherit from base classes
- [x] Created `DerivedVectorParameter` class:
  - [x] `compute()` returns numpy arrays
  - [x] `output_shape` attribute for vector-valued derived parameters (required)
  - [x] Shape and type validation on every compute
  - [x] Full integration with ParameterBank
- [x] 13 comprehensive tests - all passing

##### Phase 6: Constraints with Vectors ✅ COMPLETE
- [x] Verified constraint system works with vector parameters:
  - [x] Constraints can access vector parameters via ParameterSet
  - [x] Element-wise constraints work (e.g., `np.all(p["vec"] < 5.0)`)
  - [x] Norm constraints work (e.g., `np.linalg.norm(p["vec"]) < threshold`)
  - [x] Cross-parameter vector constraints work
  - [x] Dot product and angle constraints work
  - [x] Multiple constraints on same vector work
  - [x] Rejection sampling works correctly with vector constraints
- [x] 12 comprehensive tests demonstrating constraint patterns - all passing
- [x] No helper functions needed - constraints work naturally via lambda functions

##### Phase 7: Testing
- [x] Add comprehensive tests:
  - [x] `test_vector_parameter.py` - 24 tests covering:
    - Basic vector parameter functionality
    - Uniform and multivariate normal sampling
    - Element-wise and uniform ranges
    - Validation and error handling
    - Copy and property access
  - [ ] `test_vector_sampling.py` - Sampling with vector parameters in ParameterBank
  - [ ] `test_vector_constraints.py` - Constraints on vector parameters
  - [ ] `test_mixed_parameters.py` - Banks with both scalar and vector params
  - [ ] `test_vector_conversions.py` - Conversion between representations
  - [ ] Edge cases: empty vectors, high-dimensional arrays, etc.

##### Phase 8: Documentation
- [x] Document vector parameter usage:
  - [x] Add docstrings to `IndependentVectorParameter` class
  - [ ] Update `ParameterBank` docstrings for vector support
  - [x] Create example: `examples/vector_parameters.py`
  - [ ] Add section to README for vector parameters
  - [ ] Update API documentation

#### Module Structure ✅ COMPLETE
- [x] Consolidated `vector_parameter.py` into `parameters.py`
  - All parameter classes now in single module
  - Cleaner import structure
  - Better code organization

#### Key Design Questions ✅ RESOLVED
- [x] **Theta representation**: Flat concatenation (implemented)
  - Vector parameters are flattened into theta array
  - Unflattening reconstructs original shapes
- [x] **DataFrame representation**: Object dtype (implemented)
  - Vectors stored as numpy arrays in object dtype columns
  - Preserves array structure without expansion
- [ ] **Constraint syntax**: How should users write constraints on vectors?
  - Direct array access or helper functions?
- [ ] **Sampling efficiency**: Should we vectorize sampling for vector parameters?
- [ ] **Backward compatibility**: Ensure existing scalar-only code still works

#### Potential Challenges
- [ ] Memory efficiency for high-dimensional vector parameters
- [ ] Maintaining type safety with mixed scalar/vector parameters
- [ ] DataFrame serialization/deserialization with vectors
- [ ] Constraint checking performance with many vector parameters
- [ ] Clear error messages when shapes don't match

## Notes
- Maintain backward compatibility throughout all changes
- Update documentation as features are implemented
- Run full test suite after each major change
- Consider semantic versioning for releases
