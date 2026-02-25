"""Tests for constraints with vector parameters."""

import numpy as np
import pytest

from jscip import (
    IndependentScalarParameter,
    IndependentVectorParameter,
    ParameterBank,
)


def test_vector_element_constraint():
    """Test constraint that checks individual vector elements."""
    vec = IndependentVectorParameter(value=[1.0, 2.0, 3.0], is_sampled=True, range=(0.0, 10.0))

    # Constraint: all elements must be less than 5
    def all_elements_less_than_5(p):
        return np.all(p["vec"] < 5.0)

    bank = ParameterBank(parameters={"vec": vec}, constraints=[all_elements_less_than_5])

    # Sample with constraint
    sample = bank.sample()
    assert "vec" in sample
    assert np.all(sample["vec"] < 5.0)


def test_vector_norm_constraint():
    """Test constraint based on vector norm."""
    vec = IndependentVectorParameter(value=[1.0, 1.0], is_sampled=True, range=(-10.0, 10.0))

    # Constraint: L2 norm must be less than 5
    def norm_less_than_5(p):
        return np.linalg.norm(p["vec"]) < 5.0

    bank = ParameterBank(parameters={"vec": vec}, constraints=[norm_less_than_5])

    sample = bank.sample()
    assert np.linalg.norm(sample["vec"]) < 5.0


def test_vector_sum_constraint():
    """Test constraint on sum of vector elements."""
    vec = IndependentVectorParameter(value=[1.0, 2.0, 3.0], is_sampled=True, range=(0.0, 5.0))

    # Constraint: sum of elements must be less than 10
    def sum_less_than_10(p):
        return np.sum(p["vec"]) < 10.0

    bank = ParameterBank(parameters={"vec": vec}, constraints=[sum_less_than_10])

    sample = bank.sample()
    assert np.sum(sample["vec"]) < 10.0


def test_cross_parameter_vector_constraint():
    """Test constraint involving multiple parameters including vectors."""
    scalar = IndependentScalarParameter(value=2.0, is_sampled=True, range=(1.0, 5.0))
    vec = IndependentVectorParameter(value=[1.0, 1.0], is_sampled=True, range=(0.0, 10.0))

    # Constraint: scalar * norm(vec) < 20
    def product_constraint(p):
        return p["scalar"] * np.linalg.norm(p["vec"]) < 20.0

    bank = ParameterBank(
        parameters={"scalar": scalar, "vec": vec}, constraints=[product_constraint]
    )

    sample = bank.sample()
    assert sample["scalar"] * np.linalg.norm(sample["vec"]) < 20.0


def test_vector_dot_product_constraint():
    """Test constraint based on dot product of two vectors."""
    vec1 = IndependentVectorParameter(value=[1.0, 0.0], is_sampled=True, range=(-5.0, 5.0))
    vec2 = IndependentVectorParameter(value=[0.0, 1.0], is_sampled=True, range=(-5.0, 5.0))

    # Constraint: dot product must be positive (vectors not orthogonal)
    def dot_product_positive(p):
        return np.dot(p["vec1"], p["vec2"]) > 0.1

    bank = ParameterBank(
        parameters={"vec1": vec1, "vec2": vec2}, constraints=[dot_product_positive]
    )

    sample = bank.sample()
    assert np.dot(sample["vec1"], sample["vec2"]) > 0.1


def test_vector_angle_constraint():
    """Test constraint based on angle between vectors."""
    vec1 = IndependentVectorParameter(value=[1.0, 0.0], is_sampled=True, range=(-10.0, 10.0))
    vec2 = IndependentVectorParameter(value=[1.0, 1.0], is_sampled=True, range=(-10.0, 10.0))

    # Constraint: angle between vectors must be less than 90 degrees
    def acute_angle(p):
        v1 = p["vec1"]
        v2 = p["vec2"]
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return False
        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        return cos_angle > 0  # cos(90°) = 0

    bank = ParameterBank(parameters={"vec1": vec1, "vec2": vec2}, constraints=[acute_angle])

    sample = bank.sample()
    v1 = sample["vec1"]
    v2 = sample["vec2"]
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    assert cos_angle > 0


def test_vector_range_constraint():
    """Test constraint that limits vector to a specific range."""
    vec = IndependentVectorParameter(value=[5.0, 5.0, 5.0], is_sampled=True, range=(0.0, 10.0))

    # Constraint: all elements between 3 and 7
    def elements_in_range(p):
        return np.all((p["vec"] >= 3.0) & (p["vec"] <= 7.0))

    bank = ParameterBank(parameters={"vec": vec}, constraints=[elements_in_range])

    sample = bank.sample()
    assert np.all((sample["vec"] >= 3.0) & (sample["vec"] <= 7.0))


def test_vector_monotonic_constraint():
    """Test constraint requiring vector elements to be monotonic."""
    vec = IndependentVectorParameter(value=[1.0, 2.0, 3.0], is_sampled=True, range=(0.0, 10.0))

    # Constraint: elements must be in increasing order
    def monotonic_increasing(p):
        v = p["vec"]
        return np.all(v[1:] > v[:-1])

    bank = ParameterBank(parameters={"vec": vec}, constraints=[monotonic_increasing])

    sample = bank.sample()
    v = sample["vec"]
    assert np.all(v[1:] > v[:-1])


def test_multiple_vector_constraints():
    """Test multiple constraints on the same vector."""
    vec = IndependentVectorParameter(value=[2.0, 3.0], is_sampled=True, range=(0.0, 10.0))

    # Multiple constraints
    def sum_constraint(p):
        return np.sum(p["vec"]) < 8.0

    def norm_constraint(p):
        return np.linalg.norm(p["vec"]) < 6.0

    def min_constraint(p):
        return np.min(p["vec"]) > 1.0

    bank = ParameterBank(
        parameters={"vec": vec},
        constraints=[sum_constraint, norm_constraint, min_constraint],
    )

    sample = bank.sample()
    assert np.sum(sample["vec"]) < 8.0
    assert np.linalg.norm(sample["vec"]) < 6.0
    assert np.min(sample["vec"]) > 1.0


def test_vector_constraint_rejection_sampling():
    """Test that constraint rejection sampling works with vectors."""
    vec = IndependentVectorParameter(value=[1.0, 1.0], is_sampled=True, range=(0.0, 3.0))

    # Constraint - norm must be less than 3.0 (achievable with range 0-3)
    def small_norm(p):
        return np.linalg.norm(p["vec"]) < 3.0

    bank = ParameterBank(parameters={"vec": vec}, constraints=[small_norm])

    # Sample multiple times to ensure constraint is consistently satisfied
    for _ in range(10):
        sample = bank.sample()
        assert np.linalg.norm(sample["vec"]) < 3.0


def test_vector_constraint_with_scalar():
    """Test mixed scalar and vector constraints."""
    scalar = IndependentScalarParameter(value=5.0, is_sampled=True, range=(0.0, 10.0))
    vec = IndependentVectorParameter(value=[1.0, 2.0], is_sampled=True, range=(0.0, 10.0))

    # Constraint: scalar must be greater than max element of vector
    def scalar_greater_than_max(p):
        return p["scalar"] > np.max(p["vec"])

    bank = ParameterBank(
        parameters={"scalar": scalar, "vec": vec},
        constraints=[scalar_greater_than_max],
    )

    sample = bank.sample()
    assert sample["scalar"] > np.max(sample["vec"])


def test_vector_constraint_impossible():
    """Test that impossible constraints raise an error."""
    vec = IndependentVectorParameter(value=[5.0, 5.0], is_sampled=True, range=(0.0, 10.0))

    # Impossible constraint - norm can't be negative
    def impossible(p):
        return np.linalg.norm(p["vec"]) < 0

    bank = ParameterBank(parameters={"vec": vec}, constraints=[impossible], max_attempts=10)

    with pytest.raises(RuntimeError, match="Failed to sample"):
        bank.sample()
