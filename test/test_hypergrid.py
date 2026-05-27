"""Tests for HyperGrid functionality."""

import pytest

from jscip import (
    DerivedScalarParameter,
    HyperGrid,
    IndependentScalarParameter,
    IndependentVectorParameter,
    ParameterBank,
)


class TestHyperGrid:
    """Test suite for HyperGrid class."""

    def test_basic_hypergrid_creation(self):
        """Test basic HyperGrid creation with valid parameters."""
        # Create simple parameter bank
        param1 = IndependentScalarParameter(
            value=1.0, is_sampled=True, range=(0.0, 2.0)
        )
        param2 = IndependentScalarParameter(
            value=5.0, is_sampled=True, range=(3.0, 7.0)
        )
        fixed_param = IndependentScalarParameter(value=10.0, is_sampled=False)

        bank = ParameterBank(
            {
                "param1": param1,
                "param2": param2,
                "fixed_param": fixed_param,
            }
        )

        # Create HyperGrid
        grid = HyperGrid(bank, n_points=3)

        assert grid.n_points == 3
        assert len(grid.grid_params) == 2
        assert "param1" in grid.grid_params
        assert "param2" in grid.grid_params

    def test_hypergrid_invalid_n_points(self):
        """Test HyperGrid creation with invalid n_points."""
        param = IndependentScalarParameter(
            value=1.0, is_sampled=True, range=(0.0, 2.0)
        )
        bank = ParameterBank({"param": param})

        # Test non-integer n_points
        with pytest.raises(
            ValueError, match="n_points must be a positive integer"
        ):
            HyperGrid(bank, n_points=3.5)

        # Test zero n_points
        with pytest.raises(
            ValueError, match="n_points must be a positive integer"
        ):
            HyperGrid(bank, n_points=0)

        # Test negative n_points
        with pytest.raises(
            ValueError, match="n_points must be a positive integer"
        ):
            HyperGrid(bank, n_points=-1)

    def test_hypergrid_no_sampled_parameters(self):
        """Test HyperGrid creation with no sampled parameters."""
        param1 = IndependentScalarParameter(value=1.0, is_sampled=False)
        param2 = IndependentScalarParameter(value=2.0, is_sampled=False)

        bank = ParameterBank({"param1": param1, "param2": param2})

        with pytest.raises(
            ValueError, match="No sampled scalar parameters found"
        ):
            HyperGrid(bank, n_points=3)

    def test_hypergrid_vector_parameter_rejection(self):
        """Test that vector parameters are rejected."""
        scalar_param = IndependentScalarParameter(
            value=1.0, is_sampled=True, range=(0.0, 2.0)
        )
        vector_param = IndependentVectorParameter(
            value=[1.0, 2.0], is_sampled=True, range=(0.0, 3.0)
        )

        bank = ParameterBank(
            {
                "scalar": scalar_param,
                "vector": vector_param,
            }
        )

        with pytest.raises(
            ValueError, match="Vector-valued parameters cannot be used"
        ):
            HyperGrid(bank, n_points=3)

    def test_hypergrid_missing_range(self):
        """Test HyperGrid creation with sampled parameter missing range."""
        # Create a parameter with range None, then manually set is_sampled=True
        # This bypasses the constructor validation to test our validation
        param = IndependentScalarParameter(
            value=1.0, is_sampled=False, range=None
        )
        param._is_sampled = True  # Manually set to test our validation

        bank = ParameterBank({"param": param})

        with pytest.raises(
            ValueError, match="Sampled scalar parameters must have ranges"
        ):
            HyperGrid(bank, n_points=3)

    def test_hypergrid_generate_simple(self):
        """Test simple grid generation."""
        param1 = IndependentScalarParameter(
            value=1.0, is_sampled=True, range=(0.0, 1.0)
        )
        param2 = IndependentScalarParameter(
            value=5.0, is_sampled=True, range=(5.0, 6.0)
        )

        bank = ParameterBank({"param1": param1, "param2": param2})
        grid = HyperGrid(bank, n_points=2)

        results = grid.generate()

        # Should have 2^2 = 4 points
        assert len(results) == 4

        # Check that all combinations are present
        expected_combinations = [
            (0.0, 5.0),
            (0.0, 6.0),
            (1.0, 5.0),
            (1.0, 6.0),
        ]

        actual_combinations = [
            (result["param1"], result["param2"]) for result in results
        ]

        for expected in expected_combinations:
            assert expected in actual_combinations

    def test_hypergrid_with_fixed_parameters(self):
        """Test grid generation with fixed parameters."""
        sampled_param = IndependentScalarParameter(
            value=1.0, is_sampled=True, range=(0.0, 2.0)
        )
        fixed_param = IndependentScalarParameter(value=10.0, is_sampled=False)

        bank = ParameterBank(
            {
                "sampled": sampled_param,
                "fixed": fixed_param,
            }
        )
        grid = HyperGrid(bank, n_points=3)

        results = grid.generate()

        # Should have 3 points
        assert len(results) == 3

        # Fixed parameter should be constant
        for result in results:
            assert result["fixed"] == 10.0

        # Sampled parameter should have expected values
        sampled_values = [result["sampled"] for result in results]
        expected_values = [0.0, 1.0, 2.0]  # linspace(0, 2, 3)
        assert all(
            abs(val - exp) < 1e-10
            for val, exp in zip(sampled_values, expected_values)
        )

    def test_hypergrid_with_derived_parameters(self):
        """Test grid generation with derived parameters."""

        def compute_sum(params):
            return params["x"] + params["y"]

        x = IndependentScalarParameter(
            value=1.0, is_sampled=True, range=(0.0, 2.0)
        )
        y = IndependentScalarParameter(
            value=5.0, is_sampled=True, range=(5.0, 7.0)
        )
        sum_param = DerivedScalarParameter(compute_sum)

        bank = ParameterBank(
            {
                "x": x,
                "y": y,
                "sum": sum_param,
            }
        )
        grid = HyperGrid(bank, n_points=2)

        results = grid.generate()

        # Should have 2^2 = 4 points
        assert len(results) == 4

        # Check that derived parameter is computed correctly
        for result in results:
            expected_sum = result["x"] + result["y"]
            assert abs(result["sum"] - expected_sum) < 1e-10

    def test_hypergrid_memory_estimation(self):
        """Test memory estimation functionality."""
        param = IndependentScalarParameter(
            value=1.0, is_sampled=True, range=(0.0, 1.0)
        )
        bank = ParameterBank({"param": param})

        # Small grid
        grid = HyperGrid(bank, n_points=10)
        memory_mb = grid._estimate_memory_usage()
        assert memory_mb < 1  # Should be very small

        # Large grid (many parameters, many points)
        large_bank = ParameterBank(
            {
                f"param{i}": IndependentScalarParameter(
                    value=1.0, is_sampled=True, range=(0.0, 1.0)
                )
                for i in range(5)
            }
        )
        large_grid = HyperGrid(large_bank, n_points=10)
        large_memory_mb = large_grid._estimate_memory_usage()
        assert large_memory_mb > memory_mb  # Should be larger

    def test_hypergrid_memory_warning(self, caplog):
        """Test that memory warning is logged for large grids."""
        # Create a simple grid and test the warning mechanism
        bank = ParameterBank(
            {
                "param1": IndependentScalarParameter(
                    value=1.0, is_sampled=True, range=(0.0, 1.0)
                ),
                "param2": IndependentScalarParameter(
                    value=0.5, is_sampled=True, range=(0.0, 1.0)
                ),
            }
        )
        grid = HyperGrid(bank, n_points=5)

        # Test the warning by temporarily patching the memory estimation
        with caplog.at_level("WARNING"):
            # Patch the memory estimation to return a large value
            original_estimate = grid._estimate_memory_usage
            grid._estimate_memory_usage = lambda: 150.0  # Force warning

            # Just call generate to trigger the warning check
            try:
                grid.generate()
            except Exception:
                pass  # Don't care if it fails, we just want the warning
            finally:
                grid._estimate_memory_usage = original_estimate

        # Should have generated a warning about large memory usage
        assert any(
            "Large grid detected" in record.message
            for record in caplog.records
        )

    def test_hypergrid_parameter_ordering(self):
        """Test that generated ParameterSets maintain proper ordering."""
        param1 = IndependentScalarParameter(
            value=1.0, is_sampled=True, range=(0.0, 1.0)
        )
        param2 = IndependentScalarParameter(
            value=2.0, is_sampled=True, range=(1.0, 2.0)
        )
        param3 = IndependentScalarParameter(value=3.0, is_sampled=False)

        bank = ParameterBank(
            {
                "param2": param2,  # Note: intentionally out of alphabetical order
                "param1": param1,
                "param3": param3,
            }
        )
        grid = HyperGrid(bank, n_points=2)

        results = grid.generate()

        # Check that all results have the same ordering as the bank
        for result in results:
            assert list(result.keys()) == bank.names

    def test_hypergrid_single_parameter(self):
        """Test HyperGrid with only one sampled parameter."""
        param = IndependentScalarParameter(
            value=1.0, is_sampled=True, range=(0.0, 3.0)
        )
        bank = ParameterBank({"param": param})
        grid = HyperGrid(bank, n_points=4)

        results = grid.generate()

        # Should have 4 points
        assert len(results) == 4

        # Values should be [0.0, 1.0, 2.0, 3.0]
        expected_values = [0.0, 1.0, 2.0, 3.0]
        actual_values = [result["param"] for result in results]

        for expected, actual in zip(expected_values, actual_values):
            assert abs(actual - expected) < 1e-10

    def test_hypergrid_repr(self):
        """Test HyperGrid string representation."""
        param = IndependentScalarParameter(
            value=1.0, is_sampled=True, range=(0.0, 1.0)
        )
        bank = ParameterBank({"param": param})
        grid = HyperGrid(bank, n_points=5)

        repr_str = repr(grid)
        assert "HyperGrid" in repr_str
        assert "n_points=5" in repr_str
        assert "grid_parameters=['param']" in repr_str
