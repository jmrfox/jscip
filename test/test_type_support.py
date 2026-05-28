"""Tests for type support in parameter classes."""

import pytest
import numpy as np
from jscip import (
    IndependentScalarParameter,
    IndependentVectorParameter,
    DerivedScalarParameter,
    ParameterBank,
)


class TestTypeSupport:
    """Test type support functionality across all parameter classes."""

    def test_scalar_parameter_auto_type_detection(self):
        """Test automatic type detection for scalar parameters."""
        # Test int detection
        param_int = IndependentScalarParameter(value=42)
        assert param_int.param_type == "int"
        assert param_int.value == 42
        assert isinstance(param_int.value, int)

        # Test float detection
        param_float = IndependentScalarParameter(value=3.14)
        assert param_float.param_type == "float"
        assert param_float.value == 3.14
        assert isinstance(param_float.value, float)

        # Test bool detection
        param_bool = IndependentScalarParameter(value=True)
        assert param_bool.param_type == "bool"
        assert param_bool.value is True
        assert isinstance(param_bool.value, bool)

    def test_scalar_parameter_explicit_type_override(self):
        """Test explicit type override for scalar parameters."""
        # Override to int
        param = IndependentScalarParameter(value=3.7, param_type="int")
        assert param.param_type == "int"
        assert param.value == 3
        assert isinstance(param.value, int)

        # Override to float
        param = IndependentScalarParameter(value=42, param_type="float")
        assert param.param_type == "float"
        assert param.value == 42.0
        assert isinstance(param.value, float)

        # Override to bool
        param = IndependentScalarParameter(value=1, param_type="bool")
        assert param.param_type == "bool"
        assert param.value is True
        assert isinstance(param.value, bool)

    def test_scalar_parameter_value_setter_type_enforcement(self):
        """Test type enforcement in value setter."""
        param = IndependentScalarParameter(value=1, param_type="int")

        # Setting value should apply type constraint
        param.value = 3.7
        assert param.value == 3
        assert isinstance(param.value, int)

        param.value = "42"
        assert param.value == 42
        assert isinstance(param.value, int)

    def test_scalar_parameter_sampling_type_enforcement(self):
        """Test type enforcement during sampling."""
        param = IndependentScalarParameter(
            value=1, is_sampled=True, range=(0, 10), param_type="int"
        )

        # Sample should return int values
        sample = param.sample()
        assert isinstance(sample, int)

        # Multiple samples should all be ints
        samples = param.sample(size=5)
        assert all(isinstance(s, (int, np.integer)) for s in samples)

    def test_vector_parameter_auto_type_detection(self) -> None:
        """Test automatic type detection for vector parameters."""
        # Test int detection
        param_int = IndependentVectorParameter(value=[1, 2, 3])
        assert param_int.param_type == "int"
        assert all(isinstance(x, (int, np.integer)) for x in param_int.value)

        # Test float detection
        param_float = IndependentVectorParameter(value=[1.1, 2.2, 3.3])
        assert param_float.param_type == "float"
        assert all(isinstance(x, float) for x in param_float.value)

        # Test bool detection
        param_bool = IndependentVectorParameter(value=[True, False, True])
        assert param_bool.param_type == "bool"
        assert all(isinstance(x, (bool, np.bool_)) for x in param_bool.value)

    def test_vector_parameter_explicit_type_override(self):
        """Test explicit type override for vector parameters."""
        # Override to int
        param = IndependentVectorParameter(value=[1.7, 2.8, 3.9], param_type="int")
        assert param.param_type == "int"
        assert param.value.tolist() == [1, 2, 3]
        assert all(isinstance(x, (int, np.integer)) for x in param.value)

        # Override to float
        param = IndependentVectorParameter(value=[1, 2, 3], param_type="float")
        assert param.param_type == "float"
        assert param.value.tolist() == [1.0, 2.0, 3.0]
        assert all(isinstance(x, float) for x in param.value)

        # Override to bool
        param = IndependentVectorParameter(value=[1, 0, 2], param_type="bool")
        assert param.param_type == "bool"
        assert param.value.tolist() == [True, False, True]
        assert all(isinstance(x, (bool, np.bool_)) for x in param.value)

    def test_vector_parameter_value_setter_type_enforcement(self):
        """Test type enforcement in vector value setter."""
        param = IndependentVectorParameter(value=[1, 2, 3], param_type="int")

        # Setting value should apply type constraint
        param.value = [1.7, 2.8, 3.9]
        assert param.value.tolist() == [1, 2, 3]
        assert all(isinstance(x, (int, np.integer)) for x in param.value)

    def test_vector_parameter_sampling_type_enforcement(self):
        """Test type enforcement during vector sampling."""
        param = IndependentVectorParameter(
            value=[1, 2], is_sampled=True, range=(0, 10), param_type="int"
        )

        # Sample should return int values
        sample = param.sample()
        assert all(isinstance(x, (int, np.integer)) for x in sample)

        # Multiple samples should all be ints
        samples = param.sample(size=3)
        assert all(
            isinstance(x, (int, np.integer)) for sample in samples for x in sample
        )

    def test_derived_parameter_type_support(self):
        """Test type support for derived parameters."""

        def compute_sum(params):
            return params["a"] + params["b"]

        # Test with explicit type
        derived = DerivedScalarParameter(compute_sum, param_type="int")
        assert derived.param_type == "int"

        # Test type enforcement in compute
        from jscip import ParameterSet

        params = ParameterSet({"a": 1.7, "b": 2.3})
        result = derived.compute(params)
        assert result == 4  # 1.7 + 2.3 = 4.0, converted to int
        assert isinstance(result, int)

    def test_derived_parameter_no_type_constraint(self):
        """Test derived parameter without type constraint."""

        def compute_ratio(params):
            return params["a"] / params["b"]

        derived = DerivedScalarParameter(compute_ratio)
        assert derived.param_type is None

        # Should return the raw result without coercion
        from jscip import ParameterSet

        params = ParameterSet({"a": 1.0, "b": 3.0})
        result = derived.compute(params)
        # 1/3 is not an integer — confirms no int coercion is applied
        assert abs(result - 1.0 / 3.0) < 1e-12
        assert result != int(result)

    def test_hypergrid_type_consistency(self):
        """Test that hypergrid generation respects type constraints."""
        # Create parameters with different types
        param_int = IndependentScalarParameter(
            value=1, is_sampled=True, range=(0, 5), grid_points=3, param_type="int"
        )
        param_float = IndependentScalarParameter(
            value=1.0, is_sampled=True, range=(0, 5), grid_points=2, param_type="float"
        )
        param_bool = IndependentScalarParameter(
            value=True, is_sampled=True, range=(0, 1), grid_points=2, param_type="bool"
        )

        bank = ParameterBank(
            {
                "int_param": param_int,
                "float_param": param_float,
                "bool_param": param_bool,
            }
        )

        grid = bank.compute_hypergrid()

        # Check that all grid points have correct types
        for point in grid:
            assert isinstance(point["int_param"], int)
            assert isinstance(point["float_param"], float)
            assert isinstance(point["bool_param"], bool)

    def test_parameter_copy_preserves_type(self):
        """Test that parameter copy preserves type configuration."""
        # Test scalar parameter
        original = IndependentScalarParameter(
            value=1.7, param_type="int", range=(0, 10)
        )
        copied = original.copy()
        assert copied.param_type == "int"
        assert copied.value == 1
        assert isinstance(copied.value, int)

        # Test vector parameter
        original = IndependentVectorParameter(
            value=[1.7, 2.8], param_type="int", range=(0, 10)
        )
        copied = original.copy()
        assert copied.param_type == "int"
        assert copied.value.tolist() == [1, 2]
        assert all(isinstance(x, (int, np.integer)) for x in copied.value)

        # Test derived parameter
        def compute_func(params):
            return params["x"] * 2

        original = DerivedScalarParameter(compute_func, param_type="float")
        copied = original.copy()
        assert copied.param_type == "float"

    def test_type_constraint_errors(self):
        """Test error handling for invalid type constraints."""
        # Test invalid param_type in constructor
        with pytest.raises(
            ValueError, match="param_type must be 'int', 'float', 'bool'"
        ):
            IndependentScalarParameter(value=1, param_type="invalid")

        # Test invalid param_type setter
        param = IndependentScalarParameter(value=1)
        with pytest.raises(
            ValueError, match="param_type must be 'int', 'float', 'bool'"
        ):
            param.param_type = "invalid"

    def test_type_detection_edge_cases(self):
        """Test edge cases in type detection."""
        # Test string that can be converted to int
        param = IndependentScalarParameter(value="42")
        assert param.param_type == "int"
        assert param.value == 42

        # Test string that can be converted to float
        param = IndependentScalarParameter(value="3.14")
        assert param.param_type == "float"
        assert param.value == 3.14

        # Test string that cannot be converted
        with pytest.raises(ValueError, match="Cannot detect type from value"):
            IndependentScalarParameter(value="invalid")

    def test_mixed_type_vector_parameter(self):
        """Test vector parameter with mixed initial types."""
        # Should auto-detect from first element and convert all
        param = IndependentVectorParameter(value=[1, 2.5, True])
        assert param.param_type == "int"  # From first element
        assert param.value.tolist() == [1, 2, 1]  # All converted to int
