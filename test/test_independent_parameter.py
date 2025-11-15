import numpy as np
import pytest
from scipy import stats

from jscip import IndependentParameter, ParameterSet


def test_value_and_range_validation():
    with pytest.raises(ValueError):
        IndependentParameter(value="a")
    with pytest.raises(ValueError):
        IndependentParameter(value=1.0, is_sampled=True, range=(0.0,))
    with pytest.raises(ValueError):
        IndependentParameter(value=1.0, is_sampled=True, range=(0.0, "b"))
    with pytest.raises(ValueError):
        IndependentParameter(value=2.0, is_sampled=True, range=(0.0, 1.0))


def test_sample_returns_value_when_not_sampled():
    p = IndependentParameter(value=3.14, is_sampled=False)
    assert p.sample() == pytest.approx(3.14)


def test_sample_within_range_when_sampled():
    p = IndependentParameter(value=0.5, is_sampled=True, range=(0.0, 1.0))
    samples = p.sample(size=1000)
    assert np.min(samples) >= 0.0
    assert np.max(samples) <= 1.0


def test_copy_round_trip():
    p = IndependentParameter(value=0.5, is_sampled=True, range=(0.0, 1.0))
    q = p.copy()
    assert q.value == pytest.approx(p.value)
    assert q.range == p.range


class DummyUnit:
    def __init__(self, name: str = "u"):
        self.name = name

    def __rmul__(self, other):
        # Mimic a simple quantity object for testing: (magnitude, unit_name)
        return (float(other), self.name)


def test_sample_with_unit_optional_return_unit():
    u = DummyUnit("m")
    p = IndependentParameter(value=1.5, is_sampled=False, unit=u)
    numeric = p.sample()
    with_unit = p.sample(return_unit=True)
    assert isinstance(numeric, float)
    assert with_unit == (pytest.approx(1.5), "m")


def test_sample_array_with_unit_optional_return_unit():
    u = DummyUnit("m")
    p = IndependentParameter(value=0.0, is_sampled=True, range=(0.0, 1.0), unit=u)
    samples_with_unit = p.sample(size=10, return_unit=True)
    # Expect an array-like of quantity tuples
    assert len(samples_with_unit) == 10
    magnitudes = [q[0] for q in samples_with_unit]
    units = {q[1] for q in samples_with_unit}
    assert all(0.0 <= m <= 1.0 for m in magnitudes)
    assert units == {"m"}


def test_param_type_inferred_from_value():
    p_float = IndependentParameter(value=1.0)
    p_int = IndependentParameter(value=1)
    assert p_float._type is float
    assert p_int._type is int


def test_integer_parameter_sampling_returns_integers():
    p = IndependentParameter(value=1, is_sampled=True, range=(0, 10))
    samples = p.sample(size=50)
    assert samples.dtype == int
    assert all(isinstance(x, (int, np.integer)) for x in samples)


def test_integer_parameter_rejects_incompatible_distributions():
    with pytest.raises(ValueError):
        IndependentParameter(
            value=0,
            is_sampled=True,
            range=(-5, 5),
            distribution={"kind": "normal", "loc": 0.0, "scale": 1.0},
        )
    with pytest.raises(ValueError):
        IndependentParameter(
            value=1,
            is_sampled=True,
            range=(1, 10),
            distribution={"kind": "lognormal", "s": 0.5, "scale": 1.0},
        )


def test_normal_distribution_config_samples_reasonably():
    p = IndependentParameter(
        value=0.0,
        is_sampled=True,
        distribution={"kind": "normal", "loc": 0.0, "scale": 1.0},
    )
    samples = p.sample(size=1000)
    assert samples.shape == (1000,)
    # Rough sanity: mean should be near 0 for standard normal
    assert abs(np.mean(samples)) < 0.2


def test_exponential_and_gamma_distributions_for_floats():
    # Exponential
    p_exp = IndependentParameter(
        value=1.0,
        is_sampled=True,
        distribution={"kind": "exponential", "scale": 2.0},
    )
    samples_exp = p_exp.sample(size=100)
    assert samples_exp.shape == (100,)
    assert np.all(samples_exp >= 0.0)

    # Gamma
    p_gamma = IndependentParameter(
        value=1.0,
        is_sampled=True,
        distribution={"kind": "gamma", "a": 2.0, "scale": 1.0},
    )
    samples_gamma = p_gamma.sample(size=100)
    assert samples_gamma.shape == (100,)
    assert np.all(samples_gamma >= 0.0)


def test_discrete_distributions_for_integers():
    # Poisson
    p_poisson = IndependentParameter(
        value=1,
        is_sampled=True,
        distribution={"kind": "poisson", "mu": 3.0},
    )
    s_poisson = p_poisson.sample(size=50)
    assert s_poisson.dtype == int

    # Binomial
    p_binom = IndependentParameter(
        value=0,
        is_sampled=True,
        distribution={"kind": "binomial", "n": 10, "p": 0.5},
    )
    s_binom = p_binom.sample(size=50)
    assert s_binom.dtype == int
    assert np.all((s_binom >= 0) & (s_binom <= 10))

    # Discrete uniform over [1, 3]
    p_disc = IndependentParameter(
        value=1,
        is_sampled=True,
        distribution={"kind": "discrete_uniform", "low": 1, "high": 3},
    )
    s_disc = p_disc.sample(size=50)
    assert s_disc.dtype == int
    assert set(np.unique(s_disc)).issubset({1, 2, 3})

    # Geometric
    p_geom = IndependentParameter(
        value=1,
        is_sampled=True,
        distribution={"kind": "geometric", "p": 0.4},
    )
    s_geom = p_geom.sample(size=50)
    assert s_geom.dtype == int


def test_bernoulli_distribution_config_samples_0_or_1():
    p = IndependentParameter(
        value=0.0,
        is_sampled=True,
        distribution={"kind": "bernoulli", "p": 0.3},
    )
    samples = p.sample(size=1000)
    assert set(np.unique(samples)).issubset({0, 1})


def test_user_supplied_frozen_distribution_is_used():
    frozen = stats.norm(loc=1.0, scale=0.1)
    p = IndependentParameter(value=1.0, is_sampled=True, distribution=frozen)
    samples = p.sample(size=500)
    assert samples.shape == (500,)
    assert 0.5 < np.mean(samples) < 1.5


def test_invalid_distribution_kind_raises():
    with pytest.raises(ValueError):
        IndependentParameter(
            value=0.0,
            is_sampled=True,
            distribution={"kind": "not-a-real-kind"},
        )


def test_missing_required_distribution_params_raise():
    # lognormal requires parameter 's'
    with pytest.raises(ValueError):
        IndependentParameter(
            value=1.0,
            is_sampled=True,
            distribution={"kind": "lognormal"},
        )
    # bernoulli requires parameter 'p'
    with pytest.raises(ValueError):
        IndependentParameter(
            value=0.0,
            is_sampled=True,
            distribution={"kind": "bernoulli"},
        )


def test_parameterset_satisfies():
    ps = ParameterSet({"a": 1.0, "b": 2.0})
    assert ps.satisfies(lambda p: p["a"] < p["b"])
    with pytest.raises(ValueError):
        ps.satisfies("not a function")
