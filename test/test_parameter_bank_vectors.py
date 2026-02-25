"""Tests for ParameterBank with vector parameters."""

import numpy as np
import pytest

from jscip import (
    DerivedScalarParameter,
    IndependentScalarParameter,
    IndependentVectorParameter,
    ParameterBank,
    ParameterSet,
)


def test_parameter_bank_with_vector_parameter():
    """Test creating a ParameterBank with a vector parameter."""
    scalar = IndependentScalarParameter(value=1.0, is_sampled=True, range=(0.0, 2.0))
    vector = IndependentVectorParameter(
        value=[1.0, 2.0, 3.0], is_sampled=True, range=(0.0, 5.0)
    )

    bank = ParameterBank(parameters={"scalar": scalar, "vector": vector})

    assert len(bank) == 2
    assert "scalar" in bank
    assert "vector" in bank
    assert bank.vector_names == ["vector"]


def test_parameter_bank_vector_names():
    """Test vector_names property."""
    v1 = IndependentVectorParameter(value=[1.0, 2.0], is_sampled=False)
    v2 = IndependentVectorParameter(value=[3.0, 4.0, 5.0], is_sampled=True, range=(0.0, 10.0))
    s1 = IndependentScalarParameter(value=1.0, is_sampled=True, range=(0.0, 2.0))

    bank = ParameterBank(parameters={"v1": v1, "v2": v2, "s1": s1})

    assert set(bank.vector_names) == {"v1", "v2"}
    assert bank.sampled == ["v2", "s1"]


def test_parameter_bank_lower_upper_bounds_with_vectors():
    """Test lower_bounds and upper_bounds with vector parameters."""
    scalar = IndependentScalarParameter(value=1.0, is_sampled=True, range=(0.0, 2.0))
    vector = IndependentVectorParameter(
        value=[1.0, 2.0], is_sampled=True, range=(0.0, 5.0)
    )

    bank = ParameterBank(parameters={"scalar": scalar, "vector": vector})

    # Should have 3 bounds total: 1 scalar + 2 vector elements
    assert len(bank.lower_bounds) == 3
    assert len(bank.upper_bounds) == 3
    assert np.allclose(bank.lower_bounds, [0.0, 0.0, 0.0])
    assert np.allclose(bank.upper_bounds, [2.0, 5.0, 5.0])


def test_parameter_bank_bounds_with_elementwise_vector_ranges():
    """Test bounds with element-wise vector ranges."""
    vector = IndependentVectorParameter(
        value=[1.0, 2.0, 3.0],
        is_sampled=True,
        range=(np.array([0.0, 1.0, 2.0]), np.array([5.0, 6.0, 7.0])),
    )

    bank = ParameterBank(parameters={"vector": vector})

    assert np.allclose(bank.lower_bounds, [0.0, 1.0, 2.0])
    assert np.allclose(bank.upper_bounds, [5.0, 6.0, 7.0])


def test_parameter_bank_mixed_scalar_vector():
    """Test ParameterBank with mixed scalar and vector parameters."""
    s1 = IndependentScalarParameter(value=1.0, is_sampled=True, range=(0.0, 2.0))
    s2 = IndependentScalarParameter(value=3.0, is_sampled=False)
    v1 = IndependentVectorParameter(
        value=[1.0, 2.0], is_sampled=True, range=(0.0, 5.0)
    )
    v2 = IndependentVectorParameter(value=[3.0, 4.0, 5.0], is_sampled=False)

    bank = ParameterBank(parameters={"s1": s1, "s2": s2, "v1": v1, "v2": v2})

    assert len(bank) == 4
    assert bank.names == ["s1", "s2", "v1", "v2"]
    assert bank.sampled == ["s1", "v1"]
    assert bank.vector_names == ["v1", "v2"]


def test_parameter_bank_copy_with_vectors():
    """Test that copy() works with vector parameters."""
    vector = IndependentVectorParameter(
        value=[1.0, 2.0, 3.0], is_sampled=True, range=(0.0, 5.0)
    )
    scalar = IndependentScalarParameter(value=1.0, is_sampled=True, range=(0.0, 2.0))

    bank = ParameterBank(parameters={"vector": vector, "scalar": scalar})
    copy = bank.copy()

    assert len(copy) == 2
    assert "vector" in copy
    assert "scalar" in copy
    assert copy.vector_names == ["vector"]

    # Verify it's a true copy
    assert copy["vector"] is not bank["vector"]
    assert copy["scalar"] is not bank["scalar"]


def test_parameter_bank_getitem_with_vectors():
    """Test __getitem__ returns correct parameter types."""
    vector = IndependentVectorParameter(value=[1.0, 2.0], is_sampled=False)
    scalar = IndependentScalarParameter(value=1.0, is_sampled=False)

    bank = ParameterBank(parameters={"vector": vector, "scalar": scalar})

    assert isinstance(bank["vector"], IndependentVectorParameter)
    assert isinstance(bank["scalar"], IndependentScalarParameter)
    assert np.allclose(bank["vector"].value, [1.0, 2.0])
    assert bank["scalar"].value == 1.0


def test_parameter_bank_with_derived_and_vectors():
    """Test ParameterBank with derived parameters computed from vectors."""
    position = IndependentVectorParameter(
        value=[1.0, 2.0, 3.0], is_sampled=True, range=(0.0, 10.0)
    )

    def compute_distance(ps):
        """Compute distance from origin."""
        return np.linalg.norm(ps["position"])

    distance = DerivedScalarParameter(compute_distance)

    bank = ParameterBank(parameters={"position": position, "distance": distance})

    assert len(bank) == 2
    assert bank.vector_names == ["position"]
    assert bank.sampled == ["position"]


def test_parameter_bank_empty_with_vectors():
    """Test creating an empty ParameterBank and adding vector parameters."""
    bank = ParameterBank()

    assert len(bank) == 0
    assert bank.vector_names == []

    # Add a vector parameter
    vector = IndependentVectorParameter(value=[1.0, 2.0], is_sampled=False)
    bank.add_parameter("vec", vector)

    assert len(bank) == 1
    assert bank.vector_names == ["vec"]


def test_parameter_bank_invalid_parameter_type():
    """Test that ParameterBank rejects invalid parameter types."""
    with pytest.raises(ValueError, match="must be an instance of"):
        ParameterBank(parameters={"invalid": "not a parameter"})


def test_parameter_bank_multiple_vectors_different_shapes():
    """Test ParameterBank with multiple vectors of different shapes."""
    v1 = IndependentVectorParameter(
        value=[1.0, 2.0], is_sampled=True, range=(0.0, 5.0)
    )
    v2 = IndependentVectorParameter(
        value=[3.0, 4.0, 5.0], is_sampled=True, range=(0.0, 10.0)
    )
    v3 = IndependentVectorParameter(
        value=[6.0, 7.0, 8.0, 9.0], is_sampled=True, range=(0.0, 15.0)
    )

    bank = ParameterBank(parameters={"v1": v1, "v2": v2, "v3": v3})

    # Total bounds: 2 + 3 + 4 = 9
    assert len(bank.lower_bounds) == 9
    assert len(bank.upper_bounds) == 9
    assert np.allclose(bank.lower_bounds, [0.0] * 9)
    assert np.allclose(bank.upper_bounds, [5.0, 5.0, 10.0, 10.0, 10.0, 15.0, 15.0, 15.0, 15.0])
