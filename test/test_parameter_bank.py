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
        theta_sampling=False,
    )
    return bank


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
