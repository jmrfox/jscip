import numpy as np
import pandas as pd
import pytest

from jscip import DerivedParameter, IndependentParameter, ParameterBank, ParameterSet


def make_bank():
    p1 = IndependentParameter(value=0.5, is_sampled=True, range=(0.0, 1.0))
    p2 = IndependentParameter(value=2.0, is_sampled=False)
    d1 = DerivedParameter(lambda ps: ps["p1"] * ps["p2"])  # product
    bank = ParameterBank(
        parameters={"p1": p1, "p2": p2, "d1": d1},
        constraints=[lambda ps: ps["p1"] >= 0.2],
        vector_mode=False,
    )
    return bank


class DummyUnit:
    def __init__(self, name: str = "u"):
        self.name = name

    def __rmul__(self, other):
        return (float(other), self.name)


def test_sample_single_parameterset():
    bank = make_bank()
    ps = bank.sample()
    assert isinstance(ps, ParameterSet)
    assert set(ps.index) == {"p1", "p2", "d1"}
    assert 0.0 <= ps["p1"] <= 1.0
    assert ps["p2"] == pytest.approx(2.0)
    assert ps["d1"] == pytest.approx(ps["p1"] * ps["p2"])  # derived correct


def test_sample_dataframe_multiple():
    bank = make_bank()
    df = bank.sample(size=10)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (10, 3)
    assert set(df.columns) == {"p1", "p2", "d1"}


def test_log_prob_inside_and_outside():
    bank = make_bank()
    inside = ParameterSet({"p1": 0.3, "p2": 2.0, "d1": 0.6})
    outside = ParameterSet({"p1": 1.5, "p2": 2.0, "d1": 3.0})
    assert bank.log_prob(inside) == 0.0
    assert bank.log_prob(outside) == -np.inf


def test_instances_to_dataframe():
    bank = make_bank()
    samples = [
        ParameterSet({"p1": 0.25, "p2": 2.0, "d1": 0.5}),
        ParameterSet({"p1": 0.75, "p2": 2.0, "d1": 1.5}),
    ]
    df = bank.instances_to_dataframe(samples)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 3)


def test_get_default_values_with_units_returns_unitful_parameterset():
    # Only p1 will have a unit; p2 and d1 remain numeric
    u = DummyUnit("m")
    p1 = IndependentParameter(value=0.5, is_sampled=True, range=(0.0, 1.0), unit=u)
    p2 = IndependentParameter(value=2.0, is_sampled=False)
    d1 = DerivedParameter(lambda ps: ps["p1"] * ps["p2"])
    bank = ParameterBank(parameters={"p1": p1, "p2": p2, "d1": d1})

    ps = bank.get_default_values(with_units=True)
    assert isinstance(ps, ParameterSet)
    assert ps["p1"][1] == "m"  # quantity-like tuple (value, unit)
    assert isinstance(ps["p2"], float)
    assert isinstance(ps["d1"], float)


def test_sample_with_units_single_parameterset():
    u = DummyUnit("m")
    p1 = IndependentParameter(value=0.5, is_sampled=True, range=(0.0, 1.0), unit=u)
    p2 = IndependentParameter(value=2.0, is_sampled=False)
    d1 = DerivedParameter(lambda ps: ps["p1"] * ps["p2"])
    bank = ParameterBank(parameters={"p1": p1, "p2": p2, "d1": d1})

    ps = bank.sample(with_units=True)
    assert isinstance(ps, ParameterSet)
    assert ps["p1"][1] == "m"
    # d1 is computed from numeric values and should remain numeric
    assert isinstance(ps["d1"], float)


def test_vector_aliases_match_theta_methods():
    """The vector-based helper methods should behave consistently in vector_mode."""
    bank = make_bank()
    # Enable vector_mode so that vectors represent only sampled parameters.
    bank.vector_mode = True
    # Use a deterministic ParameterSet for round-trip tests.
    ps = ParameterSet({"p1": 0.4, "p2": 2.0, "d1": 0.8})

    # instance_to_vector should produce the parameter vector over sampled parameters
    vector_from_instance = bank.instance_to_vector(ps)
    assert vector_from_instance.shape == (len(bank.sampled),)

    df = bank.instances_to_dataframe([ps])
    vector_from_df = bank.dataframe_to_vector(df)
    assert vector_from_df.shape == (1, len(bank.sampled))

    # vector_to_instance should invert instance_to_vector when vector_mode is True
    recovered_from_vector = bank.vector_to_instance(vector_from_instance)
    assert isinstance(recovered_from_vector, ParameterSet)
    # Only sampled parameters are guaranteed to match from the vector
    for name in bank.sampled:
        assert recovered_from_vector[name] == pytest.approx(ps[name])


def test_summary(capsys):
    """summary returns a string."""
    bank = make_bank()
    summary_text = bank.summary()
    assert isinstance(summary_text, str)
    # basic sanity checks on content
    assert "ParameterBank:" in summary_text
    assert "Constraints:" in summary_text
    assert "p1" in summary_text
