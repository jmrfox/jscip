"""Tests for the new hypergrid integration in ParameterBank."""

import pytest
import numpy as np
from jscip import (
    IndependentScalarParameter,
    ParameterBank,
    DerivedScalarParameter,
)


class TestHypergridIntegration:
    """Test the integrated hypergrid functionality in ParameterBank."""

    def test_basic_linear_grid(self):
        """Test basic linear grid generation."""
        param1 = IndependentScalarParameter(
            value=1.0, is_sampled=True, range=(0.0, 2.0), grid_points=3
        )
        param2 = IndependentScalarParameter(
            value=5.0, is_sampled=True, range=(4.0, 6.0), grid_points=2
        )

        bank = ParameterBank({"param1": param1, "param2": param2})
        grid = bank.compute_hypergrid()

        assert len(grid) == 6  # 3 * 2 = 6 points

        # Check first point (0.0, 4.0)
        assert grid[0]["param1"] == 0.0
        assert grid[0]["param2"] == 4.0

        # Check last point (2.0, 6.0)
        assert grid[-1]["param1"] == 2.0
        assert grid[-1]["param2"] == 6.0

        # Check that all points are present
        expected_values = [
            (0.0, 4.0),
            (0.0, 6.0),
            (1.0, 4.0),
            (1.0, 6.0),
            (2.0, 4.0),
            (2.0, 6.0),
        ]
        actual_values = [(point["param1"], point["param2"]) for point in grid]
        assert set(actual_values) == set(expected_values)

    def test_logarithmic_grid(self):
        """Test logarithmic grid generation."""
        param = IndependentScalarParameter(
            value=10.0,
            is_sampled=True,
            range=(1.0, 100.0),
            grid_points=3,
            grid_scale="log",
        )

        bank = ParameterBank({"param": param})
        grid = bank.compute_hypergrid()

        assert len(grid) == 3
        # Logarithmic spacing: [1.0, 10.0, 100.0]
        expected_values = [1.0, 10.0, 100.0]
        actual_values = [point["param"] for point in grid]
        np.testing.assert_array_almost_equal(actual_values, expected_values)

    def test_explicit_point_list(self):
        """Test grid generation with explicit point lists."""
        param = IndependentScalarParameter(
            value=5.0,
            is_sampled=True,
            range=(0.0, 10.0),
            grid_points=[1.0, 5.0, 9.0],
        )

        bank = ParameterBank({"param": param})
        grid = bank.compute_hypergrid()

        assert len(grid) == 3
        expected_values = [1.0, 5.0, 9.0]
        actual_values = [point["param"] for point in grid]
        assert actual_values == expected_values

    def test_mixed_grid_configurations(self):
        """Test mixing different grid configurations."""
        param_linear = IndependentScalarParameter(
            value=1.0,
            is_sampled=True,
            range=(0.0, 2.0),
            grid_points=2,
            grid_scale="linear",
        )
        param_log = IndependentScalarParameter(
            value=10.0,
            is_sampled=True,
            range=(1.0, 100.0),
            grid_points=2,
            grid_scale="log",
        )
        param_explicit = IndependentScalarParameter(
            value=5.0,
            is_sampled=True,
            range=(0.0, 10.0),
            grid_points=[3.0, 7.0],
        )

        bank = ParameterBank(
            {
                "linear": param_linear,
                "log": param_log,
                "explicit": param_explicit,
            }
        )
        grid = bank.compute_hypergrid()

        assert len(grid) == 8  # 2 * 2 * 2 = 8 points

        # Check that all combinations are present
        linear_values = [0.0, 2.0]
        log_values = [1.0, 100.0]
        explicit_values = [3.0, 7.0]

        for point in grid:
            assert point["linear"] in linear_values
            assert point["log"] in log_values
            assert point["explicit"] in explicit_values

    def test_grid_with_derived_parameters(self):
        """Test grid generation with derived parameters."""
        param1 = IndependentScalarParameter(
            value=2.0, is_sampled=True, range=(1.0, 3.0), grid_points=2
        )
        param2 = IndependentScalarParameter(
            value=3.0, is_sampled=True, range=(2.0, 4.0), grid_points=2
        )

        def compute_sum(params):
            return params["param1"] + params["param2"]

        derived = DerivedScalarParameter(compute_sum)

        bank = ParameterBank({"param1": param1, "param2": param2, "sum": derived})

        grid = bank.compute_hypergrid()

        assert len(grid) == 4

        # Check that derived parameters are computed correctly
        for point in grid:
            expected_sum = point["param1"] + point["param2"]
            assert point["sum"] == expected_sum

    def test_grid_with_fixed_parameters(self):
        """Test that fixed parameters are included with current values."""
        param_sampled = IndependentScalarParameter(
            value=1.0, is_sampled=True, range=(0.0, 2.0), grid_points=2
        )
        param_fixed = IndependentScalarParameter(value=5.0, is_sampled=False)

        bank = ParameterBank({"sampled": param_sampled, "fixed": param_fixed})

        grid = bank.compute_hypergrid()

        assert len(grid) == 2
        for point in grid:
            assert point["fixed"] == 5.0

    def test_error_no_grid_points(self):
        """Test error when no parameters have grid_points specified."""
        param = IndependentScalarParameter(value=1.0, is_sampled=True, range=(0.0, 2.0))
        bank = ParameterBank({"param": param})

        with pytest.raises(
            ValueError,
            match="No sampled parameters have grid_points specified",
        ):
            bank.compute_hypergrid()

    def test_error_vector_parameter_sampling(self):
        """Test error when vector parameters are marked for sampling."""
        from jscip import IndependentVectorParameter

        vec_param = IndependentVectorParameter(
            value=[1.0, 2.0], is_sampled=True, range=(0.0, 10.0)
        )
        scalar_param = IndependentScalarParameter(
            value=1.0, is_sampled=True, range=(0.0, 2.0), grid_points=2
        )

        bank = ParameterBank({"vec": vec_param, "scalar": scalar_param})

        with pytest.raises(
            ValueError,
            match="Vector-valued parameters cannot be used in hypergrid",
        ):
            bank.compute_hypergrid()

    def test_error_log_scale_negative_range(self):
        """Test error when using log scale with negative range."""
        with pytest.raises(
            ValueError,
            match="Range must be positive for logarithmic grid scaling",
        ):
            IndependentScalarParameter(
                value=1.0,
                is_sampled=True,
                range=(-1.0, 2.0),
                grid_points=3,
                grid_scale="log",
            )

    def test_error_log_scale_negative_explicit_points(self):
        """Test error when using log scale with negative explicit points."""
        with pytest.raises(
            ValueError,
            match="Grid points must be positive for logarithmic scaling",
        ):
            IndependentScalarParameter(
                value=1.0,
                is_sampled=True,
                range=(0.1, 10.0),
                grid_points=[-1.0, 1.0, 10.0],
                grid_scale="log",
            )

    def test_error_invalid_grid_scale(self):
        """Test error when using invalid grid_scale."""
        with pytest.raises(ValueError, match="grid_scale must be 'linear' or 'log'"):
            IndependentScalarParameter(
                value=1.0,
                is_sampled=True,
                range=(0.0, 2.0),
                grid_points=3,
                grid_scale="invalid",
            )

    def test_error_invalid_grid_points_type(self):
        """Test error when using invalid grid_points type."""
        with pytest.raises(
            ValueError,
            match="grid_points sequence must contain only numeric values",
        ):
            IndependentScalarParameter(
                value=1.0,
                is_sampled=True,
                range=(0.0, 2.0),
                grid_points="invalid",  # type: ignore[arg-type]
            )

    def test_error_grid_points_int_missing_range(self):
        """Test error when using int grid_points without range."""
        with pytest.raises(
            ValueError, match="If is_sampled is True, range must be a tuple"
        ):
            IndependentScalarParameter(value=1.0, is_sampled=True, grid_points=3)

    def test_parameter_copy_preserves_grid_config(self):
        """Test that parameter copy preserves grid configuration."""
        original = IndependentScalarParameter(
            value=1.0,
            is_sampled=True,
            range=(1.0, 10.0),  # Positive range for log scale
            grid_points=5,
            grid_scale="log",
        )

        copied = original.copy()

        assert copied.grid_points == original.grid_points
        assert copied.grid_scale == original.grid_scale
        assert copied.value == original.value
        assert copied.range == original.range
        assert copied.is_sampled == original.is_sampled

    def test_grid_point_setters(self):
        """Test grid point property setters."""
        param = IndependentScalarParameter(
            value=1.0, is_sampled=True, range=(1.0, 10.0)
        )

        # Test grid_points setter
        param.grid_points = 3
        assert param.grid_points == 3

        # Test grid_scale setter
        param.grid_scale = "log"
        assert param.grid_scale == "log"

        # Test invalid grid_scale
        with pytest.raises(ValueError, match="grid_scale must be 'linear' or 'log'"):
            param.grid_scale = "invalid"  # type: ignore[assignment]
