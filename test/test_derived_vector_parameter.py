"""Tests for DerivedVectorParameter class."""

import numpy as np
import pytest

from jscip import (
    DerivedVectorParameter,
    IndependentScalarParameter,
    IndependentVectorParameter,
    ParameterBank,
    ParameterSet,
)


def test_derived_vector_parameter_init():
    """Test DerivedVectorParameter initialization."""

    def compute_vector(p):
        return np.array([p["x"], p["y"], p["z"]])

    param = DerivedVectorParameter(function=compute_vector, output_shape=(3,))

    assert param.output_shape == (3,)
    assert param.shape == (3,)
    assert not param.is_sampled
    assert callable(param.function)


def test_derived_vector_parameter_invalid_output_shape():
    """Test that invalid output_shape raises ValueError."""

    def compute_vector(p):
        return np.array([1.0, 2.0])

    # Not a tuple
    with pytest.raises(ValueError, match="output_shape must be a tuple"):
        DerivedVectorParameter(function=compute_vector, output_shape=[2])

    # Non-positive dimension
    with pytest.raises(ValueError, match="output_shape must contain positive integers"):
        DerivedVectorParameter(function=compute_vector, output_shape=(0,))

    # Non-integer dimension
    with pytest.raises(ValueError, match="output_shape must contain positive integers"):
        DerivedVectorParameter(function=compute_vector, output_shape=(2.5,))


def test_derived_vector_parameter_compute():
    """Test computing derived vector values."""

    def velocity(p):
        """Compute velocity from position."""
        return np.array([p["vx"], p["vy"]])

    param = DerivedVectorParameter(function=velocity, output_shape=(2,))

    params = ParameterSet({"vx": 3.0, "vy": 4.0})
    result = param.compute(params)

    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)
    np.testing.assert_array_equal(result, [3.0, 4.0])


def test_derived_vector_parameter_shape_validation():
    """Test that output shape is validated."""

    def wrong_shape(p):
        return np.array([1.0, 2.0, 3.0])  # Returns (3,) but expects (2,)

    param = DerivedVectorParameter(function=wrong_shape, output_shape=(2,))
    params = ParameterSet({"x": 1.0})

    with pytest.raises(ValueError, match="output shape .* does not match expected shape"):
        param.compute(params)


def test_derived_vector_parameter_non_array_output():
    """Test that non-array output raises ValueError."""

    def returns_scalar(p):
        return 5.0  # Returns scalar instead of array

    param = DerivedVectorParameter(function=returns_scalar, output_shape=(1,))
    params = ParameterSet({"x": 1.0})

    with pytest.raises(ValueError, match="must return a numpy array"):
        param.compute(params)


def test_derived_vector_parameter_in_parameter_bank():
    """Test using DerivedVectorParameter in a ParameterBank."""
    x = IndependentScalarParameter(value=1.0, is_sampled=True, range=(0.0, 10.0))
    y = IndependentScalarParameter(value=2.0, is_sampled=True, range=(0.0, 10.0))

    def position_vector(p):
        return np.array([p["x"], p["y"]])

    pos = DerivedVectorParameter(function=position_vector, output_shape=(2,))

    bank = ParameterBank(parameters={"x": x, "y": y, "position": pos})

    assert "position" in bank.names
    assert bank["position"] == pos


def test_derived_vector_parameter_in_sample():
    """Test that derived vector parameters are computed during sampling."""
    x = IndependentScalarParameter(value=3.0, is_sampled=False)
    y = IndependentScalarParameter(value=4.0, is_sampled=False)

    def magnitude_vector(p):
        """Return a 1D vector containing the magnitude."""
        return np.array([np.sqrt(p["x"] ** 2 + p["y"] ** 2)])

    mag = DerivedVectorParameter(function=magnitude_vector, output_shape=(1,))

    bank = ParameterBank(parameters={"x": x, "y": y, "magnitude": mag})

    sample = bank.sample()

    assert "magnitude" in sample
    assert isinstance(sample["magnitude"], np.ndarray)
    assert sample["magnitude"].shape == (1,)
    np.testing.assert_almost_equal(sample["magnitude"][0], 5.0)


def test_derived_vector_parameter_with_vector_input():
    """Test derived vector parameter that depends on another vector parameter."""
    vec = IndependentVectorParameter(value=[1.0, 2.0, 3.0], is_sampled=False, range=(0.0, 10.0))

    def normalized(p):
        """Normalize the input vector."""
        v = p["vec"]
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    norm_vec = DerivedVectorParameter(function=normalized, output_shape=(3,))

    bank = ParameterBank(parameters={"vec": vec, "normalized": norm_vec})

    sample = bank.sample()

    assert "normalized" in sample
    assert isinstance(sample["normalized"], np.ndarray)
    assert sample["normalized"].shape == (3,)
    # Check that it's normalized
    np.testing.assert_almost_equal(np.linalg.norm(sample["normalized"]), 1.0)


def test_derived_vector_parameter_copy():
    """Test copying a DerivedVectorParameter."""

    def compute_vec(p):
        return np.array([p["a"], p["b"]])

    param = DerivedVectorParameter(function=compute_vec, output_shape=(2,))
    param_copy = param.copy()

    assert param_copy.output_shape == param.output_shape
    assert param_copy.function == param.function
    assert param_copy.is_sampled == param.is_sampled
    assert param_copy is not param


def test_derived_vector_parameter_repr():
    """Test string representation."""

    def my_function(p):
        return np.array([1.0, 2.0])

    param = DerivedVectorParameter(function=my_function, output_shape=(2,))
    repr_str = repr(param)

    assert "DerivedVectorParameter" in repr_str
    assert "my_function" in repr_str
    assert "(2,)" in repr_str


def test_derived_vector_multidimensional():
    """Test derived vector parameter with 2D output."""

    def create_matrix(p):
        """Create a 2x2 matrix from parameters."""
        return np.array([[p["a"], p["b"]], [p["c"], p["d"]]])

    param = DerivedVectorParameter(function=create_matrix, output_shape=(2, 2))

    params = ParameterSet({"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0})
    result = param.compute(params)

    assert result.shape == (2, 2)
    np.testing.assert_array_equal(result, [[1.0, 2.0], [3.0, 4.0]])


def test_derived_vector_parameter_not_callable():
    """Test that non-callable function raises ValueError."""
    with pytest.raises(ValueError, match="Function must be callable"):
        DerivedVectorParameter(function="not_a_function", output_shape=(2,))


def test_derived_vector_parameter_invalid_parameters():
    """Test compute with invalid parameters."""

    def compute_vec(p):
        return np.array([p["x"]])

    param = DerivedVectorParameter(function=compute_vec, output_shape=(1,))

    with pytest.raises(ValueError, match="Parameters must be an instance of ParameterSet"):
        param.compute({"x": 1.0})  # Dict instead of ParameterSet
