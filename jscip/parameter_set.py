"""ParameterSet class for jscip.

This module defines the ParameterSet class, which represents a single
configuration of parameters.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ParameterSet(pd.Series):
    """A single parameter configuration with scalar and/or vector values.

    This is a thin wrapper around ``pandas.Series`` used to represent a single
    instance of parameters, typically produced by sampling a ``ParameterBank``.
    It can store both scalar values (from ``IndependentScalarParameter`` or
    ``DerivedScalarParameter``) and vector values (from ``IndependentVectorParameter``
    as numpy arrays). It preserves the canonical parameter ordering maintained
    by the bank when reindexed via ``ParameterBank.order``.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __repr__(self) -> str:
        return f"ParameterSet({super().__repr__()})"

    def satisfies(self, constraint: Callable[[ParameterSet], bool]) -> bool:
        """Evaluate a boolean constraint on this instance.

        Args:
            constraint: A callable ``f(ps: ParameterSet) -> bool``.

        Returns:
            bool: True if the constraint is satisfied, otherwise False.

        Raises:
            ValueError: If ``constraint`` is not callable or does not return a
                boolean-like value.
        """
        if not callable(constraint):
            raise ValueError("Constraint must be a callable function.")
        result = constraint(self)
        if not isinstance(result, (bool, np.bool_)):
            raise ValueError("Constraint function must return a boolean value.")
        return result

    def copy(self) -> ParameterSet:
        """Return a copy of this parameter set.

        Returns:
            ParameterSet: A new instance with the same values.

        Note:
            Numpy arrays are deep copied to prevent unintended mutations.
        """
        # Deep copy any numpy arrays to prevent shared references
        data = {}
        for key, value in self.items():
            if isinstance(value, np.ndarray):
                data[key] = value.copy()
            else:
                data[key] = value
        result = ParameterSet(data)
        logger.debug("Copied ParameterSet: %s", result)
        return result

    def reindex(self, new_index: Sequence[str]) -> ParameterSet:
        """Reindex this instance to a new sequence of parameter names.

        Args:
            new_index: Iterable of parameter names specifying the new order.

        Returns:
            ParameterSet: A new instance with the requested index.

        Raises:
            ValueError: If ``new_index`` is not a list or tuple.
        """
        if not isinstance(new_index, (list, tuple)):
            raise ValueError("New index must be a list or tuple of parameter names.")
        new_series = super().reindex(new_index)
        result = ParameterSet(new_series)
        logger.debug("Reindexed ParameterSet: %s", result)
        return result
