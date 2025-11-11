from importlib.metadata import PackageNotFoundError, version

from .main import (
    DerivedParameter,
    IndependentParameter,
    ParameterBank,
    ParameterSet,
)

__all__ = [
    "IndependentParameter",
    "ParameterSet",
    "DerivedParameter",
    "ParameterBank",
]

try:
    __version__ = version("jscip")
except PackageNotFoundError:
    __version__ = "0.1.0"
