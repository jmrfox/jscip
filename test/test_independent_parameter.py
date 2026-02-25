import numpy as np
import pytest

from jscip import IndependentScalarParameter, ParameterSet


def test_value_and_range_validation():
    with pytest.raises(ValueError):
        IndependentScalarParameter(value="a")
    with pytest.raises(ValueError):
        IndependentScalarParameter(value=1.0, is_sampled=True, range=(0.0,))
    with pytest.raises(ValueError):
        IndependentScalarParameter(value=1.0, is_sampled=True, range=(0.0, "b"))
    with pytest.raises(ValueError):
        IndependentScalarParameter(value=2.0, is_sampled=True, range=(0.0, 1.0))


def test_sample_returns_value_when_not_sampled():
    p = IndependentScalarParameter(value=3.14, is_sampled=False)
    assert p.sample() == pytest.approx(3.14)


def test_sample_within_range_when_sampled():
    p = IndependentScalarParameter(value=0.5, is_sampled=True, range=(0.0, 1.0))
    samples = p.sample(size=1000)
    assert np.min(samples) >= 0.0
    assert np.max(samples) <= 1.0


def test_copy_round_trip():
    p = IndependentScalarParameter(value=0.5, is_sampled=True, range=(0.0, 1.0))
    q = p.copy()
    assert q.value == pytest.approx(p.value)
    assert q.range == p.range


def test_parameterset_satisfies():
    ps = ParameterSet({"a": 1.0, "b": 2.0})
    assert ps.satisfies(lambda p: p["a"] < p["b"])
    with pytest.raises(ValueError):
        ps.satisfies("not a function")
