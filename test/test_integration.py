"""Integration tests for complete jscip workflows.

These tests verify that all components work together correctly in realistic
end-to-end scenarios.
"""

import numpy as np
import pytest

from jscip import (
    DerivedScalarParameter,
    IndependentScalarParameter,
    ParameterBank,
    ParameterSet,
)


def test_complete_workflow_basic():
    """Test a complete workflow: create, sample, convert, validate."""
    # Create parameters
    x = IndependentScalarParameter(value=1.0, is_sampled=True, range=(0.0, 2.0))
    y = IndependentScalarParameter(value=2.0, is_sampled=True, range=(1.0, 3.0))
    z = IndependentScalarParameter(value=5.0, is_sampled=False)

    # Create derived parameter
    def compute_sum(ps):
        return ps["x"] + ps["y"] + ps["z"]

    total = DerivedScalarParameter(compute_sum)

    # Create bank with theta_sampling for theta conversions
    bank = ParameterBank(
        parameters={"x": x, "y": y, "z": z, "total": total},
        constraints=[lambda ps: ps["total"] < 10.0],
        theta_sampling=True,
    )

    # Sample (returns array with theta_sampling=True)
    theta_sample = bank.sample()
    assert isinstance(theta_sample, np.ndarray)

    # Convert to full instance
    sample = bank.theta_to_instance(theta_sample)
    assert isinstance(sample, ParameterSet)
    assert "x" in sample
    assert "y" in sample
    assert "z" in sample
    assert "total" in sample
    assert sample["total"] < 10.0

    # Sample multiple
    theta_batch = bank.sample(size=10)
    assert isinstance(theta_batch, np.ndarray)
    assert theta_batch.shape == (10, 2)

    # Convert to theta
    theta = bank.instance_to_theta(sample)
    assert isinstance(theta, np.ndarray)
    assert len(theta) == 2  # Only x and y are sampled

    # Convert back
    reconstructed = bank.theta_to_instance(theta)
    assert isinstance(reconstructed, ParameterSet)
    assert reconstructed["x"] == pytest.approx(sample["x"])
    assert reconstructed["y"] == pytest.approx(sample["y"])


def test_complete_workflow_with_theta_sampling():
    """Test workflow with theta_sampling mode for optimization."""
    # Create parameters
    a = IndependentScalarParameter(value=0.0, is_sampled=True, range=(-5.0, 5.0))
    b = IndependentScalarParameter(value=0.0, is_sampled=True, range=(-5.0, 5.0))

    def objective(ps):
        return (ps["a"] - 1.0) ** 2 + (ps["b"] + 2.0) ** 2

    obj = DerivedScalarParameter(objective)

    # Create bank with theta_sampling
    bank = ParameterBank(
        parameters={"a": a, "b": b, "objective": obj},
        theta_sampling=True,
    )

    # Sample returns arrays
    theta_single = bank.sample()
    assert isinstance(theta_single, np.ndarray)
    assert theta_single.shape == (2,)

    # Sample batch
    theta_batch = bank.sample(size=5)
    assert isinstance(theta_batch, np.ndarray)
    assert theta_batch.shape == (5, 2)

    # Convert to full instance
    instance = bank.theta_to_instance(theta_single)
    assert "a" in instance
    assert "b" in instance
    assert "objective" in instance

    # Verify objective is computed correctly
    expected_obj = (instance["a"] - 1.0) ** 2 + (instance["b"] + 2.0) ** 2
    assert instance["objective"] == pytest.approx(expected_obj)


def test_complete_workflow_with_merging():
    """Test workflow involving merging multiple banks."""
    # Create first bank
    bank1 = ParameterBank(
        parameters={
            "mass": IndependentScalarParameter(1.0, is_sampled=True, range=(0.5, 2.0)),
        }
    )

    # Create second bank
    bank2 = ParameterBank(
        parameters={
            "velocity": IndependentScalarParameter(
                10.0, is_sampled=True, range=(5.0, 15.0)
            ),
        }
    )

    # Merge
    bank1.merge(bank2)

    # Add derived parameter
    def kinetic_energy(ps):
        return 0.5 * ps["mass"] * ps["velocity"] ** 2

    bank1.add_parameter("ke", DerivedScalarParameter(kinetic_energy))

    # Add constraint
    bank1.add_constraint(lambda ps: ps["ke"] < 100.0)

    # Sample and verify
    samples = bank1.sample(size=20)
    assert len(samples) == 20
    assert all(samples["ke"] < 100.0)
    assert all(samples["mass"] >= 0.5)
    assert all(samples["mass"] <= 2.0)
    assert all(samples["velocity"] >= 5.0)
    assert all(samples["velocity"] <= 15.0)


def test_complete_workflow_copy_and_modify():
    """Test workflow involving copying and modifying banks."""
    # Create original bank
    original = ParameterBank(
        parameters={
            "x": IndependentScalarParameter(1.0, is_sampled=True, range=(0.0, 2.0)),
            "y": IndependentScalarParameter(2.0, is_sampled=False),
        },
        constraints=[lambda ps: ps["x"] < 1.5],
        max_attempts=50,
    )

    # Copy
    copy = original.copy()

    # Modify copy
    copy.add_parameter(
        "z", IndependentScalarParameter(3.0, is_sampled=True, range=(2.0, 4.0))
    )
    copy.add_constraint(lambda ps: ps["z"] > 2.5)

    # Verify original unchanged
    assert "z" not in original
    assert len(original.constraints) == 1

    # Verify copy has changes
    assert "z" in copy
    assert len(copy.constraints) == 2

    # Sample from both
    orig_sample = original.sample()
    copy_sample = copy.sample()

    assert "z" not in orig_sample
    assert "z" in copy_sample
    assert copy_sample["z"] > 2.5


def test_complete_workflow_dataframe_roundtrip():
    """Test complete workflow with DataFrame conversions."""
    # Create bank with theta_sampling for theta conversions
    bank = ParameterBank(
        parameters={
            "p1": IndependentScalarParameter(1.0, is_sampled=True, range=(0.0, 2.0)),
            "p2": IndependentScalarParameter(2.0, is_sampled=True, range=(1.0, 3.0)),
            "p3": IndependentScalarParameter(3.0, is_sampled=False),
        },
        theta_sampling=True,
    )

    # Sample to theta array (theta_sampling=True returns arrays)
    theta = bank.sample(size=10)
    assert isinstance(theta, np.ndarray)
    assert theta.shape == (10, 2)  # Only p1 and p2 are sampled

    # Convert each row back to instance
    instances = [bank.theta_to_instance(row) for row in theta]

    # Verify roundtrip
    for i, instance in enumerate(instances):
        assert instance["p1"] == pytest.approx(theta[i, 0])
        assert instance["p2"] == pytest.approx(theta[i, 1])
        assert instance["p3"] == pytest.approx(3.0)  # Fixed value


def test_complete_workflow_log_prob():
    """Test complete workflow with log probability calculations."""
    # Create bank
    bank = ParameterBank(
        parameters={
            "x": IndependentScalarParameter(0.5, is_sampled=True, range=(0.0, 1.0)),
            "y": IndependentScalarParameter(0.5, is_sampled=True, range=(0.0, 1.0)),
        },
        constraints=[lambda ps: ps["x"] + ps["y"] < 1.5],
    )

    # Sample valid configurations
    samples = bank.sample(size=10)

    # Compute log probabilities
    log_probs = bank.log_prob(samples)

    # All should be 0.0 (valid) or -inf (invalid, shouldn't happen with constraints)
    assert all((lp == 0.0) or (lp == -np.inf) for lp in log_probs)
    assert all(lp == 0.0 for lp in log_probs)  # All should be valid

    # Create invalid sample
    invalid = ParameterSet({"x": 1.5, "y": 1.5})  # Outside bounds
    log_prob_invalid = bank.log_prob(invalid)
    assert log_prob_invalid == -np.inf


def test_complete_workflow_with_texnames():
    """Test workflow with TeX names for plotting."""
    # Create bank with TeX names
    bank = ParameterBank(
        parameters={
            "alpha": IndependentScalarParameter(1.0, is_sampled=True, range=(0.0, 2.0)),
            "beta": IndependentScalarParameter(2.0, is_sampled=True, range=(1.0, 3.0)),
            "gamma": IndependentScalarParameter(3.0, is_sampled=False),
        },
        texnames={
            "alpha": r"$\alpha$",
            "beta": r"$\beta$",
            "gamma": r"$\gamma$",
        },
    )

    # Get TeX names for sampled parameters
    tex_labels = bank.sampled_texnames
    assert len(tex_labels) == 2
    assert r"$\alpha$" in tex_labels
    assert r"$\beta$" in tex_labels

    # Sample and verify
    samples = bank.sample(size=5)
    assert len(samples) == 5


def test_complete_workflow_constraint_failure():
    """Test workflow when constraints cannot be satisfied."""
    # Create bank with impossible constraints
    bank = ParameterBank(
        parameters={
            "x": IndependentScalarParameter(0.5, is_sampled=True, range=(0.0, 1.0)),
        },
        constraints=[lambda ps: ps["x"] > 10.0],  # Impossible!
        max_attempts=5,
    )

    # Should raise RuntimeError
    with pytest.raises(RuntimeError, match="Failed to sample.*after 5 attempts"):
        bank.sample()


def test_complete_workflow_multidimensional_sampling():
    """Test workflow with multi-dimensional sampling."""
    # Create bank with theta_sampling
    bank = ParameterBank(
        parameters={
            "x": IndependentScalarParameter(0.5, is_sampled=True, range=(0.0, 1.0)),
            "y": IndependentScalarParameter(0.5, is_sampled=True, range=(0.0, 1.0)),
        },
        theta_sampling=True,
    )

    # Sample with multi-dimensional shape
    samples = bank.sample(size=(3, 4))
    assert samples.shape == (3, 4, 2)

    # Verify all values are within bounds
    assert np.all(samples >= 0.0)
    assert np.all(samples <= 1.0)


def test_complete_workflow_default_values():
    """Test workflow with default values and conversions."""
    # Create bank
    bank = ParameterBank(
        parameters={
            "a": IndependentScalarParameter(1.0, is_sampled=True, range=(0.0, 2.0)),
            "b": IndependentScalarParameter(2.0, is_sampled=False),
        }
    )

    # Get defaults as ParameterSet
    defaults = bank.get_default_values(return_theta=False)
    assert isinstance(defaults, ParameterSet)
    assert defaults["a"] == pytest.approx(1.0)
    assert defaults["b"] == pytest.approx(2.0)

    # Get defaults as theta array
    defaults_theta = bank.get_default_values(return_theta=True)
    assert isinstance(defaults_theta, np.ndarray)
    assert len(defaults_theta) == 1  # Only 'a' is sampled
    assert defaults_theta[0] == pytest.approx(1.0)


def test_complete_workflow_bounds_checking():
    """Test workflow with bounds checking."""
    # Create bank
    bank = ParameterBank(
        parameters={
            "x": IndependentScalarParameter(0.5, is_sampled=True, range=(0.0, 1.0)),
            "y": IndependentScalarParameter(0.5, is_sampled=True, range=(0.0, 1.0)),
        }
    )

    # Get bounds
    lower = bank.lower_bounds
    upper = bank.upper_bounds

    assert len(lower) == 2
    assert len(upper) == 2
    assert np.all(lower == np.array([0.0, 0.0]))
    assert np.all(upper == np.array([1.0, 1.0]))

    # Sample and verify all within bounds
    samples = bank.sample(size=100)
    theta = bank.dataframe_to_theta(samples)

    assert np.all(theta >= lower)
    assert np.all(theta <= upper)
