"""Parameter classes for jscip.

This module defines the core parameter types used in jscip:
- IndependentScalarParameter: Real-valued parameters with optional sampling
- DerivedScalarParameter: Computed parameters that depend on other parameters
- IndependentVectorParameter: Vector-valued parameters with optional sampling
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from .parameter_set import ParameterSet

logger = logging.getLogger(__name__)


class IndependentParameter:
    """Base class for independent parameters (scalar or vector).

    Independent parameters can be sampled from distributions or held fixed.
    They are the primary inputs to a parameter space.

    Attributes:
        is_sampled: Whether this parameter will be sampled from a distribution.
    """

    def __init__(self, is_sampled: bool = False):
        self._is_sampled = is_sampled

    @property
    def is_sampled(self) -> bool:
        """Get whether this parameter is sampled."""
        return self._is_sampled

    def sample(self, size: int | None = None):
        """Sample from the parameter's distribution.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement sample()")

    def copy(self):
        """Return a copy of this parameter.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement copy()")


class DerivedParameter:
    """Base class for derived parameters (scalar or vector).

    Derived parameters are computed from other parameters via a function.
    They are never sampled directly.

    Attributes:
        function: Callable that computes the derived value from a ParameterSet.
        is_sampled: Always False for derived parameters.
    """

    def __init__(self, function):
        if not callable(function):
            raise ValueError("Function must be callable.")
        self.function = function
        self._is_sampled = False

    @property
    def is_sampled(self) -> bool:
        """Get whether this parameter is sampled (always False)."""
        return self._is_sampled

    def compute(self, parameters):
        """Compute the derived value for a given parameter set.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement compute()")

    def copy(self):
        """Return a copy of this parameter.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement copy()")


class IndependentScalarParameter(IndependentParameter):
    """A real-valued parameter with optional uniform sampling over a range.

    This class represents a scalar numeric parameter. When `is_sampled=True`, a
    uniform distribution over `range=(low, high)` is constructed to draw
    samples; otherwise the parameter is treated as fixed at `value`.

    Attributes:
        value: Current scalar value of the parameter.
        is_sampled: Whether this parameter will be sampled from a distribution.
        range: Optional inclusive bounds `(low, high)` for the parameter.

    Raises:
        ValueError: If types are invalid, if the range is malformed, or if the
            value falls outside the provided range.
    """

    def __init__(
        self,
        value: float,
        is_sampled: bool = False,
        range: tuple[float, float] | None = None,
    ):
        super().__init__(is_sampled=is_sampled)
        self._validate_value_and_range(value, range)
        self._value = value
        self._range = range

        if is_sampled and (range is None or len(range) != 2):
            raise ValueError("If is_sampled is True, range must be a tuple of two numeric values.")

        if is_sampled and range is not None:
            self._distribution = stats.uniform(
                loc=self.range[0], scale=self.range[1] - self.range[0]
            )
        else:
            self._distribution = None
        logger.debug(
            "Initialized IndependentScalarParameter with value %s, range %s, is_sampled %s",
            value,
            range,
            is_sampled,
        )

    @property
    def value(self) -> float:
        """Get the value of the parameter."""
        return self._value

    @value.setter
    def value(self, value: float) -> None:
        self._validate_value_and_range(value, self.range)
        self._value = value
        logger.debug("Set value of IndependentScalarParameter to %s", value)

    @property
    def range(self) -> tuple[float, float] | None:
        """Get the range of the parameter."""
        return self._range

    @range.setter
    def range(self, range: tuple[float, float] | None) -> None:
        self._validate_value_and_range(self.value, range)
        self._range = range
        logger.debug("Set range of IndependentScalarParameter to %s", range)

    def __repr__(self) -> str:
        return f"IndependentScalarParameter(value={self.value}, range={self.range}, is_sampled={self.is_sampled})"

    def __str__(self) -> str:
        return self.__repr__()

    def _validate_value_and_range(self, value: float, range: tuple[float, float] | None) -> None:
        """Validate value and range consistency.

        Args:
            value: Candidate scalar value.
            range: Either `None` or a tuple `(low, high)` with numeric bounds.

        Raises:
            ValueError: If `value` is non-numeric, `range` is malformed, or
                `value` is not within the specified range.
        """
        if not isinstance(value, (int, float)):
            raise ValueError("Value must be a number.")
        if range is not None:
            if not isinstance(range, tuple) or len(range) != 2:
                raise ValueError("Range must be a tuple of two elements.")
            if not all(isinstance(x, (int, float)) for x in range):
                raise ValueError("Range must contain only numeric values.")
            if not (range[0] <= value <= range[1]):
                raise ValueError(f"Value {value} is not within the range {range}.")
        logger.debug(
            "Validated value and range of IndependentScalarParameter: value %s, range %s",
            value,
            range,
        )

    def sample(self, size: int | None = None) -> float | np.ndarray:
        """Sample from the parameter's distribution.

        If `is_sampled` is True, draws from a uniform distribution over
        `range`. Otherwise, returns the fixed `value`.

        Args:
            size: Optional number of samples. If omitted, returns a scalar.

        Returns:
            float | numpy.ndarray: A single float if `size is None`, otherwise
            a NumPy array of samples.
        """
        if self.is_sampled:
            result = self._distribution.rvs(size=size)
        else:
            result = self.value
        logger.debug("Sampled value from IndependentScalarParameter: %s", result)
        return result

    def copy(self) -> IndependentScalarParameter:
        """Return a shallow copy preserving configuration.

        Returns:
            IndependentScalarParameter: A new parameter with the same value, range,
            and sampling flag.
        """
        result = IndependentScalarParameter(
            value=self.value, is_sampled=self.is_sampled, range=self.range
        )
        logger.debug("Copied IndependentScalarParameter: %s", result)
        return result


class DerivedScalarParameter(DerivedParameter):
    """A read-only parameter computed from other parameters.

    A ``DerivedScalarParameter`` wraps a function that maps a ``ParameterSet`` to a
    scalar value. It is not sampled directly and is recomputed whenever an
    instance is formed or updated.
    """

    def __init__(self, function) -> None:
        super().__init__(function)
        logger.debug("Initialized DerivedScalarParameter with function %s", self.function)

    def __repr__(self) -> str:
        return f"DerivedScalarParameter(function={self.function.__name__})"

    def compute(self, parameters: ParameterSet) -> float:
        """Compute the derived value for a given parameter set.

        Args:
            parameters: The ``ParameterSet`` providing inputs to the function.

        Returns:
            float: The computed scalar value.

        Raises:
            ValueError: If ``parameters`` is not a ``ParameterSet`` or the
                stored function is not callable.
        """
        from .parameter_set import ParameterSet

        if not isinstance(parameters, ParameterSet):
            raise ValueError("Parameters must be an instance of ParameterSet.")
        if not callable(self.function):
            raise ValueError("Function must be callable.")
        result = self.function(parameters)
        logger.debug("Computed value of DerivedScalarParameter: %s", result)
        return result

    def copy(self):
        """Return a shallow copy preserving the underlying function.

        Returns:
            DerivedScalarParameter: A new wrapper around the same function.
        """
        result = DerivedScalarParameter(function=self.function)
        logger.debug("Copied DerivedScalarParameter: %s", result)
        return result


class DerivedVectorParameter(DerivedParameter):
    """A read-only vector parameter computed from other parameters.

    A ``DerivedVectorParameter`` wraps a function that maps a ``ParameterSet`` to a
    vector value. It is not sampled directly and is recomputed whenever an
    instance is formed or updated.

    Attributes:
        function: Callable that computes the derived vector from a ParameterSet.
        output_shape: Expected shape of the output vector (e.g., (3,) for 3D vector).
        is_sampled: Always False for derived parameters.
    """

    def __init__(self, function, output_shape: tuple[int, ...]) -> None:
        """Initialize a DerivedVectorParameter.

        Args:
            function: Callable that takes a ParameterSet and returns a numpy array.
            output_shape: Expected shape of the output (e.g., (3,) for 3D vector).

        Raises:
            ValueError: If function is not callable or output_shape is invalid.
        """
        super().__init__(function)

        if not isinstance(output_shape, tuple):
            raise ValueError("output_shape must be a tuple")
        if not all(isinstance(d, int) and d > 0 for d in output_shape):
            raise ValueError("output_shape must contain positive integers")

        self._output_shape = output_shape
        logger.debug(
            "Initialized DerivedVectorParameter with function %s, output_shape %s",
            self.function,
            output_shape,
        )

    @property
    def output_shape(self) -> tuple[int, ...]:
        """Get the expected output shape."""
        return self._output_shape

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the parameter (alias for output_shape)."""
        return self._output_shape

    def __repr__(self) -> str:
        return (
            f"DerivedVectorParameter(function={self.function.__name__}, "
            f"output_shape={self.output_shape})"
        )

    def compute(self, parameters: ParameterSet) -> np.ndarray:
        """Compute the derived vector value for a given parameter set.

        Args:
            parameters: The ``ParameterSet`` providing inputs to the function.

        Returns:
            numpy.ndarray: The computed vector value.

        Raises:
            ValueError: If ``parameters`` is not a ``ParameterSet``, the
                stored function is not callable, or the output shape doesn't
                match the expected shape.
        """
        from .parameter_set import ParameterSet

        if not isinstance(parameters, ParameterSet):
            raise ValueError("Parameters must be an instance of ParameterSet.")
        if not callable(self.function):
            raise ValueError("Function must be callable.")

        result = self.function(parameters)

        # Validate output is a numpy array
        if not isinstance(result, np.ndarray):
            raise ValueError(f"Function must return a numpy array, got {type(result)}")

        # Validate output shape
        if result.shape != self._output_shape:
            raise ValueError(
                f"Function output shape {result.shape} does not match "
                f"expected shape {self._output_shape}"
            )

        logger.debug("Computed value of DerivedVectorParameter: %s", result)
        return result

    def copy(self) -> DerivedVectorParameter:
        """Return a shallow copy preserving the underlying function.

        Returns:
            DerivedVectorParameter: A new wrapper around the same function.
        """
        result = DerivedVectorParameter(function=self.function, output_shape=self._output_shape)
        logger.debug("Copied DerivedVectorParameter: %s", result)
        return result


class IndependentVectorParameter(IndependentParameter):
    """A vector-valued parameter with optional multivariate sampling.

    This class represents an Nx1 vector parameter that can be sampled from
    multivariate distributions. The parameter can be initialized with a list
    or NumPy array and supports element-wise or uniform range specifications.

    Attributes:
        value: Current vector value as a 1D NumPy array of length N.
        shape: Tuple (N,) representing the dimensionality.
        is_sampled: Whether this parameter will be sampled from a distribution.
        range: Element-wise bounds as (low_array, high_array) or None.
        distribution: Distribution type ('uniform' or 'mvnormal').

    Raises:
        ValueError: If types are invalid, shapes are inconsistent, or values
            fall outside the provided ranges.
    """

    def __init__(
        self,
        value: list | np.ndarray,
        is_sampled: bool = False,
        range: (tuple[list | np.ndarray, list | np.ndarray] | tuple[float, float] | None) = None,
        distribution: Literal["uniform", "mvnormal"] = "uniform",
        cov: np.ndarray | None = None,
    ):
        """Initialize a IndependentVectorParameter.

        Args:
            value: Initial vector value as list or 1D array of length N.
            is_sampled: Whether to sample this parameter.
            range: Either:
                - tuple of (low, high) arrays/lists of length N for element-wise bounds
                - tuple of (low, high) floats to apply same range to all elements
                - None for no range constraints
            distribution: Distribution type - 'uniform' or 'mvnormal'.
            cov: Covariance matrix for 'mvnormal' distribution (NxN array).
                If None and distribution='mvnormal', uses identity matrix.

        Raises:
            ValueError: If value is not 1D, range shapes don't match, or
                distribution parameters are invalid.
        """
        # Call base class constructor
        super().__init__(is_sampled=is_sampled)

        # Convert value to numpy array and validate
        self._value = self._validate_and_convert_value(value)
        self._shape = self._value.shape

        # Validate and store range
        self._range = self._validate_and_convert_range(range, self._shape[0])

        # Validate distribution
        if distribution not in ("uniform", "mvnormal"):
            raise ValueError("distribution must be 'uniform' or 'mvnormal'")
        self._distribution = distribution

        # Validate is_sampled requirements
        if is_sampled and range is None:
            raise ValueError("If is_sampled is True, range must be provided.")

        # Set up distribution for sampling
        self._dist = None
        if is_sampled:
            if distribution == "uniform":
                # For uniform, we'll sample each element independently
                self._dist = None  # Will use element-wise uniform sampling
            elif distribution == "mvnormal":
                # Set up multivariate normal distribution
                if self._range is None:
                    raise ValueError("range must be provided for mvnormal distribution")
                # Use mean as midpoint of range
                mean = (self._range[0] + self._range[1]) / 2.0

                # Use provided covariance or identity
                if cov is None:
                    # Default: identity covariance (independent components)
                    cov_matrix = np.eye(self._shape[0])
                else:
                    cov_matrix = np.asarray(cov)
                    if cov_matrix.shape != (self._shape[0], self._shape[0]):
                        raise ValueError(
                            f"Covariance matrix must be {self._shape[0]}x{self._shape[0]}, "
                            f"got {cov_matrix.shape}"
                        )

                self._dist = stats.multivariate_normal(mean=mean, cov=cov_matrix)
                self._cov = cov_matrix
            else:
                raise ValueError(f"Unknown distribution: {distribution}")

        logger.debug(
            "Initialized IndependentVectorParameter with shape %s, is_sampled %s, distribution %s",
            self._shape,
            is_sampled,
            distribution,
        )

    def _validate_and_convert_value(self, value: list | np.ndarray) -> np.ndarray:
        """Validate and convert value to 1D numpy array.

        Args:
            value: Input value as list or array.

        Returns:
            1D numpy array.

        Raises:
            ValueError: If value is not 1D or contains non-numeric values.
        """
        arr = np.asarray(value, dtype=float)

        if arr.ndim != 1:
            raise ValueError(
                f"IndependentVectorParameter value must be 1D (Nx1), got shape {arr.shape}"
            )

        if arr.size == 0:
            raise ValueError("IndependentVectorParameter value cannot be empty")

        return arr

    def _validate_and_convert_range(
        self,
        range: tuple[list | np.ndarray, list | np.ndarray] | tuple[float, float] | None,
        n: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Validate and convert range specification.

        Args:
            range: Range specification (see __init__ docstring).
            n: Length of the vector.

        Returns:
            Tuple of (low_array, high_array) or None.

        Raises:
            ValueError: If range specification is invalid.
        """
        if range is None:
            return None

        if not isinstance(range, tuple) or len(range) != 2:
            raise ValueError("range must be a tuple of (low, high)")

        low, high = range

        # Case 1: Single float tuple - apply to all elements
        if isinstance(low, (int, float)) and isinstance(high, (int, float)):
            low_arr = np.full(n, float(low))
            high_arr = np.full(n, float(high))
        else:
            # Case 2: Array/list tuple - element-wise
            low_arr = np.asarray(low, dtype=float)
            high_arr = np.asarray(high, dtype=float)

            if low_arr.shape != (n,) or high_arr.shape != (n,):
                raise ValueError(
                    f"range arrays must have shape ({n},), "
                    f"got low: {low_arr.shape}, high: {high_arr.shape}"
                )

        # Validate that low <= high for all elements
        if not np.all(low_arr <= high_arr):
            raise ValueError("All low values must be <= corresponding high values")

        # Validate that current value is within range
        if not np.all((self._value >= low_arr) & (self._value <= high_arr)):
            raise ValueError(f"Value {self._value} is not within range [{low_arr}, {high_arr}]")

        return (low_arr, high_arr)

    @property
    def value(self) -> np.ndarray:
        """Get the current value of the parameter."""
        return self._value

    @value.setter
    def value(self, value: list | np.ndarray) -> None:
        """Set the value of the parameter.

        Args:
            value: New value as list or 1D array.

        Raises:
            ValueError: If value is invalid or outside range.
        """
        new_value = self._validate_and_convert_value(value)

        if new_value.shape != self._shape:
            raise ValueError(
                f"New value shape {new_value.shape} does not match parameter shape {self._shape}"
            )

        if self._range is not None:
            low, high = self._range
            if not np.all((new_value >= low) & (new_value <= high)):
                raise ValueError(f"Value {new_value} is not within range [{low}, {high}]")

        self._value = new_value
        logger.debug("Set value of IndependentVectorParameter to %s", new_value)

    @property
    def shape(self) -> tuple[int]:
        """Get the shape of the parameter."""
        return self._shape

    @property
    def range(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Get the range of the parameter."""
        return self._range

    @property
    def distribution(self) -> str:
        """Get the distribution type."""
        return self._distribution

    def __repr__(self) -> str:
        return (
            f"IndependentVectorParameter(shape={self.shape}, value={self.value}, "
            f"is_sampled={self.is_sampled}, distribution={self.distribution})"
        )

    def __str__(self) -> str:
        return self.__repr__()

    def sample(self, size: int | None = None) -> np.ndarray:
        """Sample from the parameter's distribution.

        If `is_sampled` is True, draws from the configured distribution.
        Otherwise, returns the fixed `value`.

        Args:
            size: Optional number of samples. If omitted, returns a single
                sample with shape matching the parameter shape.

        Returns:
            numpy.ndarray: If size is None, returns array of shape (N,).
                If size is provided, returns array of shape (size, N).
        """
        if not self.is_sampled:
            if size is None:
                result = self.value.copy()
            else:
                result = np.tile(self.value, (size, 1))
            logger.debug("Returned fixed value from IndependentVectorParameter: %s", result)
            return result

        if self._distribution == "uniform":
            # Sample each element independently from uniform distribution
            low, high = self._range
            if size is None:
                result = np.random.uniform(low, high)
            else:
                result = np.random.uniform(low, high, size=(size, len(low)))
        elif self._distribution == "mvnormal":
            # Sample from multivariate normal
            if size is None:
                sample = self._dist.rvs()
                # Clip to range
                low, high = self._range
                result = np.clip(sample, low, high)
            else:
                samples = self._dist.rvs(size=size)
                # Clip to range
                low, high = self._range
                result = np.clip(samples, low, high)
        else:
            raise ValueError(f"Unknown distribution: {self._distribution}")

        logger.debug("Sampled value from IndependentVectorParameter: %s", result)
        return result

    def copy(self) -> IndependentVectorParameter:
        """Return a copy preserving configuration.

        Returns:
            IndependentVectorParameter: A new parameter with the same configuration.
        """
        # Reconstruct range in original format
        if self._range is not None:
            range_copy = (self._range[0].copy(), self._range[1].copy())
        else:
            range_copy = None

        # Get covariance if mvnormal
        cov_copy = None
        if self._distribution == "mvnormal" and hasattr(self, "_cov"):
            cov_copy = self._cov.copy()

        result = IndependentVectorParameter(
            value=self.value.copy(),
            is_sampled=self.is_sampled,
            range=range_copy,
            distribution=self.distribution,
            cov=cov_copy,
        )
        logger.debug("Copied IndependentVectorParameter: %s", result)
        return result
