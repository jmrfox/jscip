import pytest

from jscip import DerivedScalarParameter, ParameterSet


def test_derived_parameter_creation():
    def square_sum(ps: ParameterSet) -> float:
        return ps["a"] ** 2 + ps["b"] ** 2

    dp = DerivedScalarParameter(square_sum)
    assert dp.function == square_sum
    assert dp.is_sampled is False


def test_derived_parameter_compute():
    def product(ps: ParameterSet) -> float:
        return ps["x"] * ps["y"]

    dp = DerivedScalarParameter(product)
    ps = ParameterSet({"x": 3.0, "y": 4.0})
    result = dp.compute(ps)
    assert result == pytest.approx(12.0)


def test_derived_parameter_invalid_function():
    with pytest.raises(ValueError, match="Function must be callable"):
        DerivedScalarParameter("not a function")


def test_derived_parameter_compute_invalid_input():
    dp = DerivedScalarParameter(lambda ps: ps["a"] + ps["b"])
    with pytest.raises(ValueError, match="must be an instance of ParameterSet"):
        dp.compute({"a": 1.0, "b": 2.0})


def test_derived_parameter_copy():
    def my_func(ps: ParameterSet) -> float:
        return ps["a"] * 2

    dp = DerivedScalarParameter(my_func)
    dp_copy = dp.copy()
    assert dp_copy.function == dp.function
    assert dp_copy.is_sampled == dp.is_sampled


def test_derived_parameter_repr():
    def named_function(ps: ParameterSet) -> float:
        return ps["x"]

    dp = DerivedScalarParameter(named_function)
    assert "named_function" in repr(dp)
