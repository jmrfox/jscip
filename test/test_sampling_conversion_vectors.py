"""Tests for sampling and conversion with vector parameters."""

import numpy as np

from jscip import (
    DerivedScalarParameter,
    IndependentScalarParameter,
    IndependentVectorParameter,
    ParameterBank,
)


def test_sample_with_vector_parameters():
    """Test sampling a ParameterBank with vector parameters."""
    scalar = IndependentScalarParameter(value=1.0, is_sampled=True, range=(0.0, 2.0))
    vector = IndependentVectorParameter(value=[1.0, 2.0], is_sampled=True, range=(0.0, 5.0))

    bank = ParameterBank(parameters={"scalar": scalar, "vector": vector})

    sample = bank.sample()

    assert "scalar" in sample
    assert "vector" in sample
    assert isinstance(sample["vector"], np.ndarray)
    assert len(sample["vector"]) == 2


def test_sample_multiple_with_vectors():
    """Test sampling multiple instances with vector parameters."""
    vector = IndependentVectorParameter(value=[1.0, 2.0, 3.0], is_sampled=True, range=(0.0, 10.0))

    bank = ParameterBank(parameters={"vector": vector})

    # When theta_sampling=False and size is int, returns DataFrame
    samples_df = bank.sample(size=10)

    assert len(samples_df) == 10
    assert "vector" in samples_df.columns
    # Each row should have a vector
    for idx in range(len(samples_df)):
        vec = samples_df.iloc[idx]["vector"]
        assert isinstance(vec, np.ndarray)
        assert len(vec) == 3


def test_instance_to_theta_with_vectors():
    """Test converting ParameterSet with vectors to theta array."""
    scalar = IndependentScalarParameter(value=1.0, is_sampled=True, range=(0.0, 2.0))
    vector = IndependentVectorParameter(value=[1.0, 2.0], is_sampled=True, range=(0.0, 5.0))

    bank = ParameterBank(parameters={"scalar": scalar, "vector": vector}, theta_sampling=False)

    sample = bank.sample()
    theta = bank.instance_to_theta(sample)

    # Should have 3 elements: 1 scalar + 2 vector elements
    assert len(theta) == 3
    assert theta[0] == sample["scalar"]
    assert np.allclose(theta[1:3], sample["vector"])


def test_theta_to_instance_with_vectors():
    """Test converting theta array back to ParameterSet with vectors."""
    scalar = IndependentScalarParameter(value=1.0, is_sampled=True, range=(0.0, 2.0))
    vector = IndependentVectorParameter(value=[1.0, 2.0], is_sampled=True, range=(0.0, 5.0))

    bank = ParameterBank(parameters={"scalar": scalar, "vector": vector}, theta_sampling=True)

    # Create a theta array: [scalar_value, vector_elem1, vector_elem2]
    theta = np.array([1.5, 3.0, 4.0])
    instance = bank.theta_to_instance(theta)

    assert instance["scalar"] == 1.5
    assert np.allclose(instance["vector"], [3.0, 4.0])


def test_roundtrip_conversion_with_vectors():
    """Test roundtrip conversion: instance -> theta -> instance."""
    vector = IndependentVectorParameter(value=[1.0, 2.0, 3.0], is_sampled=True, range=(0.0, 10.0))
    scalar = IndependentScalarParameter(value=5.0, is_sampled=True, range=(0.0, 10.0))

    bank = ParameterBank(parameters={"vector": vector, "scalar": scalar}, theta_sampling=False)

    original = bank.sample()
    theta = bank.instance_to_theta(original)

    # Now switch to theta_sampling mode for conversion back
    bank.theta_sampling = True
    recovered = bank.theta_to_instance(theta)

    assert recovered["scalar"] == original["scalar"]
    assert np.allclose(recovered["vector"], original["vector"])


def test_instance_to_theta_list_with_vectors():
    """Test converting list of ParameterSets with vectors to 2D theta array."""
    vector = IndependentVectorParameter(value=[1.0, 2.0], is_sampled=True, range=(0.0, 5.0))

    bank = ParameterBank(parameters={"vector": vector}, theta_sampling=False)

    # Sample returns DataFrame when theta_sampling=False and size is int
    # We need to convert to list of ParameterSets manually for this test
    samples_list = []
    for _ in range(5):
        sample = bank.sample()  # Single sample returns ParameterSet
        samples_list.append(sample)

    theta = bank.instance_to_theta(samples_list)

    # Should be 2D: (5 samples, 2 vector elements)
    assert theta.shape == (5, 2)


def test_get_default_values_with_vectors():
    """Test get_default_values with vector parameters."""
    vector = IndependentVectorParameter(value=[1.0, 2.0, 3.0], is_sampled=False)
    scalar = IndependentScalarParameter(value=5.0, is_sampled=False)

    bank = ParameterBank(parameters={"vector": vector, "scalar": scalar})

    defaults = bank.get_default_values(return_theta=False)

    assert defaults["scalar"] == 5.0
    assert np.allclose(defaults["vector"], [1.0, 2.0, 3.0])


def test_get_default_values_theta_with_vectors():
    """Test get_default_values returning theta with vector parameters."""
    vector = IndependentVectorParameter(value=[1.0, 2.0], is_sampled=True, range=(0.0, 5.0))
    scalar = IndependentScalarParameter(value=3.0, is_sampled=True, range=(0.0, 10.0))

    bank = ParameterBank(parameters={"scalar": scalar, "vector": vector}, theta_sampling=True)

    theta = bank.get_default_values(return_theta=True)

    # Should have 3 elements: 1 scalar + 2 vector
    assert len(theta) == 3
    assert theta[0] == 3.0
    assert np.allclose(theta[1:3], [1.0, 2.0])


def test_derived_parameter_with_vector_input():
    """Test derived parameter computed from vector parameter."""
    position = IndependentVectorParameter(value=[3.0, 4.0], is_sampled=True, range=(0.0, 10.0))

    def compute_distance(ps):
        return np.linalg.norm(ps["position"])

    distance = DerivedScalarParameter(compute_distance)

    bank = ParameterBank(parameters={"position": position, "distance": distance})

    sample = bank.sample()

    # Distance should be computed from position
    expected_distance = np.linalg.norm(sample["position"])
    assert np.isclose(sample["distance"], expected_distance)


def test_mixed_scalar_vector_sampling():
    """Test sampling with mixed scalar and vector parameters."""
    s1 = IndependentScalarParameter(value=1.0, is_sampled=True, range=(0.0, 2.0))
    s2 = IndependentScalarParameter(value=3.0, is_sampled=False)
    v1 = IndependentVectorParameter(value=[1.0, 2.0], is_sampled=True, range=(0.0, 5.0))
    v2 = IndependentVectorParameter(value=[3.0, 4.0, 5.0], is_sampled=False)

    bank = ParameterBank(parameters={"s1": s1, "s2": s2, "v1": v1, "v2": v2})

    sample = bank.sample()

    # Sampled parameters should vary
    assert "s1" in sample
    assert "v1" in sample
    # Non-sampled should be defaults
    assert sample["s2"] == 3.0
    assert np.allclose(sample["v2"], [3.0, 4.0, 5.0])


def test_theta_dimensions_with_vectors():
    """Test that theta dimensions match expected size with vectors."""
    v1 = IndependentVectorParameter(value=[1.0, 2.0], is_sampled=True, range=(0.0, 5.0))
    v2 = IndependentVectorParameter(value=[3.0, 4.0, 5.0], is_sampled=True, range=(0.0, 10.0))
    s1 = IndependentScalarParameter(value=1.0, is_sampled=True, range=(0.0, 2.0))

    bank = ParameterBank(parameters={"s1": s1, "v1": v1, "v2": v2}, theta_sampling=False)

    # Expected theta dimensions: 1 (s1) + 2 (v1) + 3 (v2) = 6
    assert len(bank.lower_bounds) == 6
    assert len(bank.upper_bounds) == 6

    sample = bank.sample()
    theta = bank.instance_to_theta(sample)
    assert len(theta) == 6


def test_elementwise_vector_range_conversion():
    """Test conversion with element-wise vector ranges."""
    vector = IndependentVectorParameter(
        value=[1.0, 2.0, 3.0],
        is_sampled=True,
        range=(np.array([0.0, 1.0, 2.0]), np.array([5.0, 6.0, 7.0])),
    )

    bank = ParameterBank(parameters={"vector": vector}, theta_sampling=True)

    theta = np.array([2.5, 3.5, 4.5])
    instance = bank.theta_to_instance(theta)

    assert np.allclose(instance["vector"], [2.5, 3.5, 4.5])
