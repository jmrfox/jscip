import numpy as np
import pandas as pd
import pytest

from jscip import (
    DerivedScalarParameter,
    IndependentScalarParameter,
    ParameterBank,
    ParameterSet,
)


def test_merge_banks():
    bank1 = ParameterBank(
        parameters={
            "a": IndependentScalarParameter(1.0, is_sampled=True, range=(0.0, 2.0)),
        }
    )
    bank2 = ParameterBank(
        parameters={
            "b": IndependentScalarParameter(3.0, is_sampled=False),
        }
    )
    bank1.merge(bank2)
    assert "a" in bank1
    assert "b" in bank1
    assert len(bank1) == 2


def test_merge_invalid_type():
    bank = ParameterBank()
    with pytest.raises(ValueError, match="must be an instance of ParameterBank"):
        bank.merge("not a bank")


def test_add_parameter():
    bank = ParameterBank()
    param = IndependentScalarParameter(5.0, is_sampled=False)
    bank.add_parameter("new_param", param)
    assert "new_param" in bank
    assert bank["new_param"] == param


def test_add_parameter_duplicate():
    bank = ParameterBank(parameters={"a": IndependentScalarParameter(1.0)})
    with pytest.raises(KeyError, match="already exists"):
        bank.add_parameter("a", IndependentScalarParameter(2.0))


def test_add_parameter_invalid_type():
    bank = ParameterBank()
    with pytest.raises(ValueError, match="must be an instance"):
        bank.add_parameter("bad", "not a parameter")


def test_add_constraint():
    bank = ParameterBank(
        parameters={
            "a": IndependentScalarParameter(1.0, is_sampled=True, range=(0.0, 2.0))
        }
    )
    constraint = lambda ps: ps["a"] > 0.5
    bank.add_constraint(constraint)
    assert constraint in bank.constraints


def test_add_constraint_invalid():
    bank = ParameterBank()
    with pytest.raises(ValueError, match="must be a callable"):
        bank.add_constraint("not callable")


def test_get_constraints():
    constraint1 = lambda ps: ps["a"] > 0
    constraint2 = lambda ps: ps["a"] < 1
    bank = ParameterBank(
        parameters={
            "a": IndependentScalarParameter(0.5, is_sampled=True, range=(0.0, 1.0))
        },
        constraints=[constraint1, constraint2],
    )
    constraints = bank.get_constraints()
    assert len(constraints) == 2
    assert constraint1 in constraints
    assert constraint2 in constraints


def test_theta_to_instance_with_theta_sampling():
    bank = ParameterBank(
        parameters={
            "a": IndependentScalarParameter(1.0, is_sampled=True, range=(0.0, 2.0)),
            "b": IndependentScalarParameter(3.0, is_sampled=False),
        },
        theta_sampling=True,
    )
    theta = np.array([1.5])
    instance = bank.theta_to_instance(theta)
    assert instance["a"] == pytest.approx(1.5)
    assert instance["b"] == pytest.approx(3.0)


def test_theta_to_instance_without_theta_sampling():
    bank = ParameterBank(
        parameters={
            "a": IndependentScalarParameter(1.0, is_sampled=True, range=(0.0, 2.0)),
            "b": IndependentScalarParameter(3.0, is_sampled=False),
        },
        theta_sampling=False,
    )
    theta = np.array([1.5, 4.0])
    instance = bank.theta_to_instance(theta)
    assert instance["a"] == pytest.approx(1.5)
    assert instance["b"] == pytest.approx(4.0)


def test_theta_to_instance_invalid_length():
    bank = ParameterBank(
        parameters={
            "a": IndependentScalarParameter(1.0, is_sampled=True, range=(0.0, 2.0)),
        },
        theta_sampling=True,
    )
    with pytest.raises(ValueError, match="does not match"):
        bank.theta_to_instance(np.array([1.0, 2.0]))


def test_dataframe_to_theta():
    bank = ParameterBank(
        parameters={
            "a": IndependentScalarParameter(1.0, is_sampled=True, range=(0.0, 2.0)),
            "b": IndependentScalarParameter(3.0, is_sampled=True, range=(2.0, 4.0)),
        }
    )
    df = pd.DataFrame({"a": [0.5, 1.0, 1.5], "b": [2.5, 3.0, 3.5]})
    theta = bank.dataframe_to_theta(df)
    assert theta.shape == (3, 2)
    assert theta[0, 0] == pytest.approx(0.5)
    assert theta[0, 1] == pytest.approx(2.5)


def test_dataframe_to_theta_invalid_input():
    bank = ParameterBank()
    with pytest.raises(ValueError, match="must be a pandas DataFrame"):
        bank.dataframe_to_theta([1, 2, 3])


def test_max_attempts_configurable():
    bank = ParameterBank(
        parameters={
            "a": IndependentScalarParameter(0.5, is_sampled=True, range=(0.0, 1.0)),
        },
        constraints=[lambda ps: ps["a"] > 10.0],  # impossible constraint
        max_attempts=5,
    )
    assert bank._max_attempts == 5
    with pytest.raises(RuntimeError, match="Failed to sample.*after 5 attempts"):
        bank.sample()


def test_max_attempts_invalid():
    with pytest.raises(ValueError, match="must be a positive integer"):
        ParameterBank(max_attempts=0)
    with pytest.raises(ValueError, match="must be a positive integer"):
        ParameterBank(max_attempts=-1)
    with pytest.raises(ValueError, match="must be a positive integer"):
        ParameterBank(max_attempts="not an int")


def test_copy_bank():
    bank = ParameterBank(
        parameters={
            "a": IndependentScalarParameter(1.0, is_sampled=True, range=(0.0, 2.0)),
        },
        constraints=[lambda ps: ps["a"] > 0.5],
        theta_sampling=True,
        texnames={"a": r"$\alpha$"},
        max_attempts=50,
    )
    bank_copy = bank.copy()
    assert len(bank_copy) == len(bank)
    assert bank_copy.theta_sampling == bank.theta_sampling
    assert bank_copy._max_attempts == bank._max_attempts
    assert "a" in bank_copy.texnames


def test_pretty_print(capsys):
    bank = ParameterBank(
        parameters={
            "a": IndependentScalarParameter(1.0, is_sampled=True, range=(0.0, 2.0)),
        }
    )
    bank.pretty_print()
    captured = capsys.readouterr()
    assert "ParameterBank" in captured.out
    assert "a:" in captured.out


def test_sample_with_multidimensional_size():
    bank = ParameterBank(
        parameters={
            "a": IndependentScalarParameter(0.5, is_sampled=True, range=(0.0, 1.0)),
        },
        theta_sampling=True,
    )
    samples = bank.sample(size=(2, 3))
    assert samples.shape == (2, 3, 1)


def test_sample_multidimensional_without_theta_sampling():
    bank = ParameterBank(
        parameters={
            "a": IndependentScalarParameter(0.5, is_sampled=True, range=(0.0, 1.0)),
        },
        theta_sampling=False,
    )
    with pytest.raises(ValueError, match="only supported for theta_sampling"):
        bank.sample(size=(2, 3))


def test_is_sampled_property():
    param = IndependentScalarParameter(1.0, is_sampled=True, range=(0.0, 2.0))
    assert param.is_sampled is True

    param2 = IndependentScalarParameter(1.0, is_sampled=False)
    assert param2.is_sampled is False

    derived = DerivedScalarParameter(lambda ps: ps["a"])
    assert derived.is_sampled is False


def test_texnames_validation():
    # Valid texnames
    bank = ParameterBank(
        parameters={
            "alpha": IndependentScalarParameter(1.0),
            "beta": IndependentScalarParameter(2.0),
        },
        texnames={"alpha": r"$\alpha$", "beta": r"$\beta$"},
    )
    assert bank.texnames["alpha"] == r"$\alpha$"

    # Invalid texnames - key not in parameters
    with pytest.raises(ValueError, match="texnames contains keys not in parameters"):
        ParameterBank(
            parameters={"a": IndependentScalarParameter(1.0)},
            texnames={"a": r"$a$", "b": r"$b$"},  # 'b' doesn't exist
        )
