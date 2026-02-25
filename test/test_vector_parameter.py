"""Tests for IndependentVectorParameter class."""

import numpy as np
import pytest

from jscip import IndependentVectorParameter


def test_create_vector_parameter_from_list():
    """Test creating a IndependentVectorParameter from a list."""
    vp = IndependentVectorParameter(value=[1.0, 2.0, 3.0])
    assert vp.shape == (3,)
    assert np.allclose(vp.value, [1.0, 2.0, 3.0])
    assert not vp.is_sampled


def test_create_vector_parameter_from_array():
    """Test creating a IndependentVectorParameter from a numpy array."""
    vp = IndependentVectorParameter(value=np.array([1.0, 2.0, 3.0]))
    assert vp.shape == (3,)
    assert np.allclose(vp.value, [1.0, 2.0, 3.0])


def test_vector_parameter_with_uniform_range():
    """Test IndependentVectorParameter with uniform range applied to all elements."""
    vp = IndependentVectorParameter(
        value=[1.0, 2.0, 3.0],
        is_sampled=True,
        range=(0.0, 5.0),
    )
    assert vp.range is not None
    low, high = vp.range
    assert np.allclose(low, [0.0, 0.0, 0.0])
    assert np.allclose(high, [5.0, 5.0, 5.0])


def test_vector_parameter_with_elementwise_range():
    """Test IndependentVectorParameter with element-wise range specification."""
    vp = IndependentVectorParameter(
        value=[1.0, 2.0, 3.0],
        is_sampled=True,
        range=([0.0, 1.0, 2.0], [2.0, 3.0, 4.0]),
    )
    assert vp.range is not None
    low, high = vp.range
    assert np.allclose(low, [0.0, 1.0, 2.0])
    assert np.allclose(high, [2.0, 3.0, 4.0])


def test_vector_parameter_invalid_shape():
    """Test that 2D arrays are rejected."""
    with pytest.raises(ValueError, match="must be 1D"):
        IndependentVectorParameter(value=np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_vector_parameter_empty_value():
    """Test that empty arrays are rejected."""
    with pytest.raises(ValueError, match="cannot be empty"):
        IndependentVectorParameter(value=[])


def test_vector_parameter_value_outside_range():
    """Test that values outside range are rejected."""
    with pytest.raises(ValueError, match="not within range"):
        IndependentVectorParameter(
            value=[1.0, 2.0, 3.0],
            range=(0.0, 2.5),
        )


def test_vector_parameter_invalid_range_shape():
    """Test that mismatched range shapes are rejected."""
    with pytest.raises(ValueError, match="range arrays must have shape"):
        IndependentVectorParameter(
            value=[1.0, 2.0, 3.0],
            is_sampled=True,
            range=([0.0, 1.0], [2.0, 3.0]),  # Only 2 elements, need 3
        )


def test_vector_parameter_low_greater_than_high():
    """Test that invalid range bounds are rejected."""
    with pytest.raises(ValueError, match="low values must be <="):
        IndependentVectorParameter(
            value=[1.0, 2.0, 3.0],
            is_sampled=True,
            range=([3.0, 2.0, 1.0], [1.0, 2.0, 3.0]),
        )


def test_vector_parameter_is_sampled_without_range():
    """Test that is_sampled=True requires a range."""
    with pytest.raises(ValueError, match="range must be provided"):
        IndependentVectorParameter(
            value=[1.0, 2.0, 3.0],
            is_sampled=True,
        )


def test_vector_parameter_sample_uniform():
    """Test sampling from uniform distribution."""
    vp = IndependentVectorParameter(
        value=[1.0, 2.0, 3.0],
        is_sampled=True,
        range=(0.0, 5.0),
        distribution="uniform",
    )

    # Single sample
    sample = vp.sample()
    assert sample.shape == (3,)
    assert np.all(sample >= 0.0)
    assert np.all(sample <= 5.0)

    # Multiple samples
    samples = vp.sample(size=10)
    assert samples.shape == (10, 3)
    assert np.all(samples >= 0.0)
    assert np.all(samples <= 5.0)


def test_vector_parameter_sample_mvnormal():
    """Test sampling from multivariate normal distribution."""
    vp = IndependentVectorParameter(
        value=[1.0, 2.0],
        is_sampled=True,
        range=(0.0, 4.0),
        distribution="mvnormal",
    )

    # Single sample
    sample = vp.sample()
    assert sample.shape == (2,)
    # Should be clipped to range
    assert np.all(sample >= 0.0)
    assert np.all(sample <= 4.0)

    # Multiple samples
    samples = vp.sample(size=100)
    assert samples.shape == (100, 2)
    assert np.all(samples >= 0.0)
    assert np.all(samples <= 4.0)


def test_vector_parameter_sample_mvnormal_with_covariance():
    """Test sampling from mvnormal with custom covariance."""
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    vp = IndependentVectorParameter(
        value=[2.0, 2.0],
        is_sampled=True,
        range=(0.0, 4.0),
        distribution="mvnormal",
        cov=cov,
    )

    samples = vp.sample(size=100)
    assert samples.shape == (100, 2)
    # Samples should be correlated due to covariance
    correlation = np.corrcoef(samples.T)[0, 1]
    # Should have positive correlation (not exact due to clipping)
    assert correlation > 0.0


def test_vector_parameter_sample_not_sampled():
    """Test that non-sampled parameters return fixed value."""
    vp = IndependentVectorParameter(value=[1.0, 2.0, 3.0], is_sampled=False)

    sample = vp.sample()
    assert np.allclose(sample, [1.0, 2.0, 3.0])

    samples = vp.sample(size=5)
    assert samples.shape == (5, 3)
    assert np.allclose(samples, [[1.0, 2.0, 3.0]] * 5)


def test_vector_parameter_invalid_distribution():
    """Test that invalid distribution types are rejected."""
    with pytest.raises(ValueError, match="must be 'uniform' or 'mvnormal'"):
        IndependentVectorParameter(
            value=[0.5, 0.5],
            is_sampled=True,
            range=(0.0, 1.0),
            distribution="normal",
        )


def test_vector_parameter_mvnormal_invalid_cov_shape():
    """Test that invalid covariance matrix shape is rejected."""
    with pytest.raises(ValueError, match="Covariance matrix must be"):
        IndependentVectorParameter(
            value=[0.5, 0.5, 0.5],
            is_sampled=True,
            range=(0.0, 1.0),
            distribution="mvnormal",
            cov=np.eye(2),  # Should be 3x3
        )


def test_vector_parameter_value_setter():
    """Test setting a new value."""
    vp = IndependentVectorParameter(value=[1.0, 2.0, 3.0], range=(0.0, 5.0))

    vp.value = [2.0, 3.0, 4.0]
    assert np.allclose(vp.value, [2.0, 3.0, 4.0])

    # Test with numpy array
    vp.value = np.array([1.5, 2.5, 3.5])
    assert np.allclose(vp.value, [1.5, 2.5, 3.5])


def test_vector_parameter_value_setter_wrong_shape():
    """Test that setting value with wrong shape is rejected."""
    vp = IndependentVectorParameter(value=[1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="does not match parameter shape"):
        vp.value = [1.0, 2.0]


def test_vector_parameter_value_setter_outside_range():
    """Test that setting value outside range is rejected."""
    vp = IndependentVectorParameter(value=[1.0, 2.0, 3.0], range=(0.0, 5.0))

    with pytest.raises(ValueError, match="not within range"):
        vp.value = [6.0, 7.0, 8.0]


def test_vector_parameter_copy():
    """Test copying a IndependentVectorParameter."""
    vp = IndependentVectorParameter(
        value=[1.0, 2.0, 3.0],
        is_sampled=True,
        range=(0.0, 5.0),
        distribution="uniform",
    )

    vp_copy = vp.copy()

    assert vp_copy is not vp
    assert np.allclose(vp_copy.value, vp.value)
    assert vp_copy.shape == vp.shape
    assert vp_copy.is_sampled == vp.is_sampled
    assert vp_copy.distribution == vp.distribution

    # Modify copy, original should be unchanged
    vp_copy.value = [2.0, 3.0, 4.0]
    assert np.allclose(vp.value, [1.0, 2.0, 3.0])


def test_vector_parameter_copy_mvnormal():
    """Test copying a IndependentVectorParameter with mvnormal distribution."""
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    vp = IndependentVectorParameter(
        value=[1.0, 2.0],
        is_sampled=True,
        range=(0.0, 4.0),
        distribution="mvnormal",
        cov=cov,
    )

    vp_copy = vp.copy()

    assert vp_copy is not vp
    assert np.allclose(vp_copy.value, vp.value)
    # Covariance should be copied
    assert hasattr(vp_copy, "_cov")


def test_vector_parameter_repr():
    """Test string representation."""
    vp = IndependentVectorParameter(
        value=[1.0, 2.0, 3.0],
        is_sampled=True,
        range=(0.0, 5.0),
    )

    repr_str = repr(vp)
    assert "IndependentVectorParameter" in repr_str
    assert "shape=(3,)" in repr_str
    assert "is_sampled=True" in repr_str


def test_vector_parameter_properties():
    """Test that all properties are read-only."""
    vp = IndependentVectorParameter(
        value=[1.0, 2.0, 3.0],
        is_sampled=True,
        range=(0.0, 5.0),
    )

    # These should all be accessible
    assert vp.shape == (3,)
    assert vp.is_sampled is True
    assert vp.range is not None
    assert vp.distribution == "uniform"

    # Shape should not be settable
    with pytest.raises(AttributeError):
        vp.shape = (4,)

    # is_sampled should not be settable
    with pytest.raises(AttributeError):
        vp.is_sampled = False


def test_vector_parameter_elementwise_range_with_arrays():
    """Test element-wise range with numpy arrays."""
    low = np.array([0.0, 1.0, 2.0])
    high = np.array([2.0, 3.0, 4.0])

    vp = IndependentVectorParameter(
        value=[1.0, 2.0, 3.0],
        is_sampled=True,
        range=(low, high),
    )

    assert vp.range is not None
    range_low, range_high = vp.range
    assert np.allclose(range_low, low)
    assert np.allclose(range_high, high)
