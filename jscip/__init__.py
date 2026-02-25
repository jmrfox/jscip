from importlib.metadata import PackageNotFoundError, version

from .parameter_bank import ParameterBank
from .parameter_set import ParameterSet
from .parameters import (
    DerivedParameter,
    DerivedScalarParameter,
    DerivedVectorParameter,
    IndependentParameter,
    IndependentScalarParameter,
    IndependentVectorParameter,
)

__all__ = [
    "IndependentScalarParameter",
    "IndependentVectorParameter",
    "ParameterSet",
    "DerivedScalarParameter",
    "ParameterBank",
    "IndependentParameter",
    "DerivedParameter",
    "DerivedVectorParameter",
]

try:
    __version__ = version("jscip")
except PackageNotFoundError:
    __version__ = "0.1.0"
