"""jscip main module.

This module provides a lightweight framework for defining and managing numerical
parameters used in scientific workflows. It supports:

- Independent, real-valued parameters with optional ranges and sampling
- Derived parameters defined by functions of other parameters
- Constraint checking on sampled configurations
- Conversions between rich parameter instances and array/dataframe forms

Key classes:

- `IndependentParameter`: A real-valued parameter with optional sampling over a
  uniform range.
- `DerivedParameter`: A read-only parameter computed from a function of a
  `ParameterSet`.
- `ParameterSet`: A single configuration (instance) of parameters.
- `ParameterBank`: A collection of parameters that can be sampled jointly,
  validated against constraints, and converted between representations.

Typical usage:

1. Create independent parameters, tagging the ones to sample with
   `is_sampled=True` and providing a `(low, high)` range.
2. Optionally define derived parameters as functions of the independent ones.
3. Add parameters to a `ParameterBank`, optionally add constraints, and sample
   instances or arrays.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
import pandas as pd
import logging
from collections.abc import Callable, Iterator, Sequence

logger = logging.getLogger(__name__)


class IndependentParameter:
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
        self._validate_value_and_range(value, range)
        self._value = value
        self._range = range
        self._is_sampled = is_sampled

        if is_sampled and (range is None or len(range) != 2):
            raise ValueError(
                "If is_sampled is True, range must be a tuple of two numeric values."
            )

        if is_sampled and range is not None:
            self._distribution = stats.uniform(
                loc=self.range[0], scale=self.range[1] - self.range[0]
            )
        else:
            self._distribution = None
        logger.debug(
            "Initialized IndependentParameter with value %s, range %s, is_sampled %s",
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
        logger.debug("Set value of IndependentParameter to %s", value)

    @property
    def range(self) -> tuple[float, float] | None:
        """Get the range of the parameter."""
        return self._range

    @range.setter
    def range(self, range: tuple[float, float] | None) -> None:
        self._validate_value_and_range(self.value, range)
        self._range = range
        logger.debug("Set range of IndependentParameter to %s", range)

    def __repr__(self) -> str:
        return f"IndependentParameter(value={self.value}, range={self.range}, is_sampled={self._is_sampled})"

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
            "Validated value and range of IndependentParameter: value %s, range %s",
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
        if self._is_sampled:
            result = self._distribution.rvs(size=size)
        else:
            result = self.value
        logger.debug("Sampled value from IndependentParameter: %s", result)
        return result

    def copy(self) -> IndependentParameter:
        """Return a shallow copy preserving configuration.

        Returns:
            IndependentParameter: A new parameter with the same value, range,
            and sampling flag.
        """
        result = IndependentParameter(
            value=self.value, is_sampled=self._is_sampled, range=self.range
        )
        logger.debug("Copied IndependentParameter: %s", result)
        return result


class ParameterSet(pd.Series):
    """A single parameter configuration with scalar values.

    This is a thin wrapper around ``pandas.Series`` used to represent a single
    instance of parameters, typically produced by sampling a ``ParameterBank``.
    It preserves the canonical parameter ordering maintained by the bank when
    reindexed via ``ParameterBank.order``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self):
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
        """
        result = ParameterSet(self.to_dict())
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


class DerivedParameter:
    """A read-only parameter computed from other parameters.

    A ``DerivedParameter`` wraps a function that maps a ``ParameterSet`` to a
    scalar value. It is not sampled directly and is recomputed whenever an
    instance is formed or updated.
    """

    def __init__(self, function: Callable[[ParameterSet], float]) -> None:
        self.function = function
        self._is_sampled = False  # Derived parameters are never considered sampled.
        if not callable(self.function):
            raise ValueError("Function must be callable.")
        logger.debug("Initialized DerivedParameter with function %s", self.function)

    def __repr__(self) -> str:
        return f"DerivedParameter(function={self.function.__name__})"

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
        if not isinstance(parameters, ParameterSet):
            raise ValueError("Parameters must be an instance of ParameterSet.")
        if not callable(self.function):
            raise ValueError("Function must be callable.")
        result = self.function(parameters)
        logger.debug("Computed value of DerivedParameter: %s", result)
        return result

    def copy(self) -> DerivedParameter:
        """Return a shallow copy preserving the underlying function.

        Returns:
            DerivedParameter: A new wrapper around the same function.
        """
        result = DerivedParameter(function=self.function)
        logger.debug("Copied DerivedParameter: %s", result)
        return result


class ParameterBank:
    """A collection of parameters with sampling, constraints, and conversions.

    The bank stores independent and derived parameters, optional constraint
    functions, and a canonical parameter order. It can sample full parameter
    instances, validate them against constraints, and convert between rich
    ``ParameterSet`` and array/dataframe representations.
    """

    def __init__(
        self,
        parameters: dict[str, IndependentParameter | DerivedParameter] | None = None,
        constraints: list[Callable[[ParameterSet], bool]] | None = None,
        theta_sampling: bool = False,
        texnames: dict[str, str] | None = None,
    ) -> None:
        self.parameters = parameters if parameters is not None else {}
        self.constraints = constraints if constraints is not None else []
        self.theta_sampling = theta_sampling
        self._max_attempts = (
            100  # Default maximum attempts for sampling with constraints
        )
        self.texnames = texnames if texnames is not None else {}

        for key, value in self.parameters.items():
            if not isinstance(value, IndependentParameter) and not isinstance(
                value, DerivedParameter
            ):
                raise ValueError(
                    f"Value for key '{key}' must be an instance of IndependentParameter or DerivedParameter."
                )

        # compute indices of sampled parameters in the canonical order
        self._refresh_sampled_indices()
        logger.debug(
            "Initialized ParameterBank with %d parameters and %d constraints",
            len(self.parameters),
            len(self.constraints),
        )

    def _refresh_sampled_indices(self) -> None:
        """Refresh cached indices for sampled parameters based on current parameters."""
        self.sampled_indices = [
            self.names.index(key)
            for key, param in self.parameters.items()
            if isinstance(param, IndependentParameter) and param._is_sampled
        ]
        logger.debug("Refreshed sampled indices: %s", self.sampled_indices)

    @property
    def names(self) -> list[str]:
        """Get the names of all parameters in the bank.
        This also defines the canonical order of the parameters."""
        return list(self.parameters.keys())

    @property
    def sampled(self) -> list[str]:
        """Get a list of all parameters that are set to be sampled."""
        return [key for key, param in self.parameters.items() if param._is_sampled]

    @property
    def lower_bounds(self) -> np.ndarray:
        """Get the lower bounds of all sampled parameters."""
        return np.array(
            [
                param.range[0]
                for key, param in self.parameters.items()
                if isinstance(param, IndependentParameter) and param._is_sampled
            ]
        )

    @property
    def upper_bounds(self) -> np.ndarray:
        """Get the upper bounds of all sampled parameters."""
        return np.array(
            [
                param.range[1]
                for key, param in self.parameters.items()
                if isinstance(param, IndependentParameter) and param._is_sampled
            ]
        )

    @property
    def sampled_texnames(self) -> list[str]:
        """Get the TeX names of all sampled parameters."""
        return [self.texnames.get(key, key) for key in self.sampled]

    def __repr__(self) -> str:
        return f"ParameterBank(parameters={self.parameters}, constraints={self.constraints})"

    def __contains__(self, key: str) -> bool:
        """Check if a parameter exists in the bank."""
        return key in self.parameters

    def __len__(self) -> int:
        """Get the number of parameters in the bank."""
        return len(self.parameters)

    def __iter__(self) -> Iterator[str]:
        """Iterate over the parameter names in the bank."""
        return iter(self.parameters)

    def __getitem__(self, key: str) -> IndependentParameter | DerivedParameter:
        """Get a parameter by its name."""
        if key in self.parameters:
            return self.parameters[key]
        else:
            raise KeyError(f"Parameter '{key}' not found in the bank.")

    def copy(self) -> ParameterBank:
        """Create a copy of the ParameterBank."""
        result = ParameterBank(
            parameters={k: v.copy() for k, v in self.parameters.items()},
            constraints=self.constraints.copy(),
            theta_sampling=self.theta_sampling,
            texnames=self.texnames.copy(),
        )
        logger.debug("Copied ParameterBank: %s", result)
        return result

    def get_value(self, key: str) -> float:
        if key in self.parameters:
            return self.parameters[key].value
        else:
            raise KeyError(f"Parameter '{key}' not found in the bank.")

    def merge(self, other: "ParameterBank") -> None:
        """Merge another ParameterBank into this one.
        If a parameter with the same name exists, it will be overwritten.
        """
        if not isinstance(other, ParameterBank):
            raise ValueError("Other must be an instance of ParameterBank.")
        for key, value in other.parameters.items():
            self.parameters[key] = value.copy()
        self.constraints.extend(other.constraints)
        self._refresh_sampled_indices()
        logger.debug("Merged ParameterBank: %s", self)

    def add_parameter(
        self, key: str, parameter: IndependentParameter | DerivedParameter
    ) -> None:
        """Add a new parameter to the bank."""
        if not isinstance(parameter, (IndependentParameter, DerivedParameter)):
            raise ValueError(
                "Parameter must be an instance of IndependentParameter or DerivedParameter."
            )
        if key in self.parameters:
            raise KeyError(f"Parameter '{key}' already exists in the bank.")
        self.parameters[key] = parameter
        logger.debug("Added parameter '%s' to ParameterBank: %s", key, self)

    def add_constraint(self, constraint: Callable[[ParameterSet], bool]) -> None:
        """Add a new constraint to the bank."""
        if not callable(constraint):
            raise ValueError("Constraint must be a callable function.")
        self.constraints.append(constraint)
        logger.debug("Added constraint '%s' to ParameterBank: %s", constraint, self)

    def get_constraints(self) -> list[Callable[[ParameterSet], bool]]:
        """Get all constraints in the bank."""
        return self.constraints

    def get_default_values(self, return_theta: bool | None = None) -> ParameterSet | np.ndarray:
        """Return default values for all parameters.

        Computes a ``ParameterSet`` by taking the current ``value`` for all
        independent parameters and computing all derived parameters from those
        values. Optionally, returns the sampled subset as a NumPy array when
        ``return_theta=True``.

        Args:
            return_theta: If True, return a 1D NumPy array of sampled parameter
                values in canonical sampled order. If False, return a full
                ``ParameterSet``. Defaults to ``self.theta_sampling``.

        Returns:
            ParameterSet | numpy.ndarray: The default instance or the sampled
            values array.

        Raises:
            ValueError: If ``return_theta`` is not a boolean.
        """
        if return_theta is None:
            return_theta = (
                self.theta_sampling
            )  # default to self.theta_sampling if not specified
        if not isinstance(return_theta, bool):
            raise ValueError("theta_array must be a boolean value.")
        p = ParameterSet(
            {
                key: param.value
                for key, param in self.parameters.items()
                if isinstance(param, IndependentParameter)
            }
        )
        logger.debug(
            "[get_default_values] Default values for all independent parameters in the bank: %s",
            p,
        )
        p = ParameterSet(
            {
                **p,
                **{
                    key: param.compute(p)
                    for key, param in self.parameters.items()
                    if isinstance(param, DerivedParameter)
                },
            }
        )
        p = self.order(p)
        logger.debug(
            "[get_default_values] Default values for all parameters in the bank: %s", p
        )
        if return_theta:
            return self.instance_to_theta(p)
        else:
            return p

    def instance_to_theta(self, input: ParameterSet | list[ParameterSet]) -> np.ndarray:
        """Convert a parameter instance (or list) to a sampled theta array.

        Args:
            input: A single ``ParameterSet`` or list of ``ParameterSet``
                instances.

        Returns:
            numpy.ndarray: 1D array for a single instance or 2D array for a
            list of instances, containing values for sampled parameters only,
            in canonical sampled order.

        Raises:
            ValueError: If ``input`` is not a ``ParameterSet`` or list thereof.
        """
        if not isinstance(input, (ParameterSet, list)):
            raise ValueError(
                "Input must be a ParameterSetInstance or a list of ParameterSetInstances."
            )
        if isinstance(input, ParameterSet):
            theta = np.array([input[key] for key in self.sampled])
            logger.debug(
                "[instance_to_theta] Converted ParameterSet to numpy array: %s", theta
            )
        else:
            # return a 2D array of shape (n_instances, n_sampled)
            theta = np.vstack(
                [
                    np.array([instance[key] for key in self.sampled])
                    for instance in input
                ]
            )
            logger.debug(
                "[instance_to_theta] Converted list of ParameterSetInstances to numpy array: %s",
                theta,
            )
        return theta

    def dataframe_to_theta(self, df: pd.DataFrame) -> np.ndarray:
        """Extract sampled parameter columns from a DataFrame as a NumPy array.

        Args:
            df: DataFrame containing sampled parameter columns.

        Returns:
            numpy.ndarray: 2D array of sampled values in canonical sampled
            order.

        Raises:
            ValueError: If ``df`` is not a pandas DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        theta = df[self.sampled].to_numpy()
        return theta

    def theta_to_instance(self, theta: np.ndarray) -> ParameterSet:
        """Convert a theta array to a parameter instance.

        When ``theta_sampling`` is True, ``theta`` must contain only sampled
        independent parameters in canonical sampled order. Otherwise, it must
        contain values for all independent parameters in canonical order.

        Args:
            theta: 1D NumPy array.

        Returns:
            ParameterSet: A full instance with derived parameters recomputed.

        Raises:
            ValueError: If shapes are inconsistent with ``theta_sampling`` or
                if ``theta`` is not a NumPy array.
        """
        if not isinstance(theta, np.ndarray):
            raise ValueError(
                "Input must be a numpy array, instead got: " + str(type(theta))
            )
        # validate length depending on theta_sampling mode
        if self.theta_sampling:
            if len(theta) != len(self.sampled):
                raise ValueError(
                    f"Array length {len(theta)} does not match number of sampled parameters {len(self.sampled)}."
                )
        else:
            if len(theta) != len(self.parameters):
                raise ValueError(
                    f"Array length {len(theta)} does not match number of parameters {len(self.parameters)}."
                )
        # theta in this case must be a 1D array
        # Start with defaults
        out = self.get_default_values(return_theta=False)
        if self.theta_sampling:
            # theta provides only sampled independent parameters
            for i, key in enumerate(self.sampled):
                out[key] = theta[i]
        else:
            # theta provides values for ALL parameters in canonical order
            if len(theta) != len(self.parameters):
                raise ValueError(
                    f"Array length {len(theta)} does not match number of parameters {len(self.parameters)}."
                )
            for i, key in enumerate(self.names):
                param = self.parameters[key]
                if isinstance(param, IndependentParameter):
                    out[key] = float(theta[i])
        # recompute derived parameters
        out = ParameterSet(
            {
                **out,
                **{
                    key: param.compute(out)
                    for key, param in self.parameters.items()
                    if isinstance(param, DerivedParameter)
                },
            }
        )
        return out

    def _sample_once(self) -> ParameterSet:
        """Sample a single full parameter set (internal).

        Samples all sampled independent parameters, computes derived values, and
        returns a ``ParameterSet`` ordered canonically.
        """
        # first, sample all independent parameters that are set to be sampled
        p = ParameterSet(
            {
                key: param.sample()
                for key, param in self.parameters.items()
                if isinstance(param, IndependentParameter)
            }
        )
        logger.debug(
            "[sample_once] Sampled values for all independent parameters in the bank: %s",
            p,
        )
        # then, compute all derived parameters based on the sampled independent parameters
        p = ParameterSet(
            {
                **p,
                **{
                    key: param.compute(p)
                    for key, param in self.parameters.items()
                    if isinstance(param, DerivedParameter)
                },
            }
        )
        logger.debug(
            "[sample_once] Sampled values for all parameters in the bank: %s", p
        )
        # put result in canonical order according to self.canonical_order
        p = self.order(p)
        return p

    def _sample_once_constrained(self) -> ParameterSet:
        """Sample a single parameter set that satisfies all constraints (internal)."""
        attempts = 0
        while attempts < self._max_attempts:
            attempts += 1
            sample = self._sample_once()
            # Check if the sample meets all constraints
            if all(sample.satisfies(c) for c in self.constraints):
                return sample
        raise RuntimeError(
            f"Failed to sample a parameter set satisfying constraints after {self._max_attempts} attempts."
        )

    def sample(self, size: int | tuple | None = None) -> ParameterSet | pd.DataFrame | np.ndarray:
        """Sample parameter sets or theta arrays.

        Args:
            size: If ``None``, returns a single instance. If ``int``, returns a
                batch. If ``tuple``, returns product size; multi-d shapes are
                only supported when ``theta_sampling`` is True.

        Returns:
            ParameterSet | pandas.DataFrame | numpy.ndarray: Depending on
            ``theta_sampling`` and ``size``.

        Raises:
            ValueError: If ``size`` has an invalid type or dimensionality.
        """
        if (
            size is not None
            and not isinstance(size, int)
            and not isinstance(size, tuple)
        ):
            raise ValueError("Size must be None, an integer, or a tuple.")
        if size is None:
            n_samples = 1
        elif isinstance(size, int):
            n_samples = size
        elif isinstance(size, tuple):
            if len(size) > 1 and not self.theta_sampling:
                raise ValueError(
                    "Multiple dimensions are only supported for theta_sampling."
                )
            if len(size) == 1:
                n_samples = size[0]
            else:
                n_samples = int(np.prod(size))

        # print("n_samples (type):", n_samples, type(n_samples))
        # print("size (type):", size, type(size))
        samples = []
        for _ in range(n_samples):
            if self.constraints:
                sample = self._sample_once_constrained()
            else:
                sample = self._sample_once()
            # add any parameters that are not sampled but are required for the model
            sample = ParameterSet(
                {
                    **sample,
                    **{
                        key: param.value
                        for key, param in self.parameters.items()
                        if not param._is_sampled
                        and not isinstance(param, DerivedParameter)
                    },
                }
            )
            samples.append(sample)
        # print("After sampling, there are", len(samples), "samples.")
        if self.theta_sampling:
            # out = np.array([self.instance_to_theta(sample) for sample in samples])
            if size is None:
                out = self.instance_to_theta(samples[0])
            elif isinstance(size, int):
                out = np.array(
                    [self.instance_to_theta(sample) for sample in samples]
                ).reshape((size, len(self.sampled)))
            elif isinstance(size, tuple):
                out = np.array(
                    [self.instance_to_theta(sample) for sample in samples]
                ).reshape(size + (len(self.sampled),))
        else:
            if size is None:
                out = samples[0]
            elif isinstance(size, int):
                out = self.instances_to_dataframe([sample for sample in samples])
            elif isinstance(size, tuple):
                out = self.instances_to_dataframe([sample for sample in samples])
        return out

    def instances_to_dataframe(self, instances: list[ParameterSet]) -> pd.DataFrame:
        """Convert a list of parameter instances to a pandas DataFrame.

        Args:
            instances: A non-empty list of ``ParameterSet`` objects.

        Returns:
            pandas.DataFrame: Rows correspond to instances; columns to
            parameters in canonical order.

        Raises:
            ValueError: If the input is not a non-empty list of ``ParameterSet``
                objects.
        """
        if not isinstance(instances, list):
            raise ValueError(
                "Instances must be a list of ParameterSetInstance objects."
            )
        if not instances:
            raise ValueError("Instances list cannot be empty.")
        if not all(isinstance(instance, ParameterSet) for instance in instances):
            raise ValueError(
                "All items in instances must be ParameterSetInstance objects."
            )
        df = pd.DataFrame([instance for instance in instances])
        df = df.astype(float)
        return df

    def log_prob(self, input: ParameterSet | pd.DataFrame | np.ndarray) -> float | np.ndarray:
        """Compute a simple log prior for samples under uniform bounds.

        Anything outside the bounds of sampled independent parameters, or
        violating constraints, receives ``-inf``; otherwise ``0``.

        Args:
            input: A ``ParameterSet``, a pandas ``DataFrame`` (rows are
                instances), or a NumPy array (1D or 2D). For arrays, the
                expected width depends on ``theta_sampling``.

        Returns:
            float | numpy.ndarray: A scalar for a single ``ParameterSet`` or a
            NumPy array of log-probabilities for batches.

        Raises:
            ValueError: If the type/shape of ``input`` is inconsistent with the
                current ``theta_sampling`` mode.
        """
        # categorize inputs
        if isinstance(input, ParameterSet):  # if a single sample, package it in a list
            samples = [input]
        elif isinstance(
            input, pd.DataFrame
        ):  # if a DataFrame, convert to list of ParameterSet instances
            samples = [ParameterSet(row) for _, row in input.iterrows()]
        elif isinstance(input, np.ndarray):  # if numpy array ...
            if input.ndim == 1:  # if 1D, treat as a single sample
                if (
                    input.shape[0] != len(self.sampled) and self.theta_sampling
                ):  # if theta_sampling is enabled, sample must match sampled parameters
                    raise ValueError(
                        f"1D numpy array must have length {len(self.sampled)} to match sampled parameters, since theta_sampling is enabled."
                    )
                elif (
                    input.shape[0] != len(self.parameters) and not self.theta_sampling
                ):  # if theta_sampling is disabled, sample must match all parameters
                    raise ValueError(
                        f"1D numpy array must have length {len(self.parameters)} to match all parameters, since theta_sampling is disabled."
                    )
                # print("Converting 1D numpy array to ParameterSet instance.")
                samples = [self.theta_to_instance(input)]  # convert to ParameterSet
            elif input.ndim == 2:  # if 2D, treat each row as a sample
                if input.shape[1] != len(self.sampled) and self.theta_sampling:
                    raise ValueError(
                        f"2D numpy array must have {len(self.sampled)} columns to match sampled parameters, since theta_sampling is enabled."
                    )
                elif input.shape[1] != len(self.parameters) and not self.theta_sampling:
                    raise ValueError(
                        f"2D numpy array must have {len(self.parameters)} columns to match all parameters, since theta_sampling is disabled."
                    )
                # print("Converting 2D numpy array to list of ParameterSet instances.")
                samples = [self.theta_to_instance(row) for row in input]
            else:
                raise ValueError("Samples must be a 1D or 2D numpy array.")
        elif not isinstance(input, list):
            raise ValueError(
                "Samples must be a list of ParameterSet instances or a numpy array."
            )

        results = np.zeros(len(samples))
        for i, sample in enumerate(samples):
            results[i] = self._log_prob_single(sample)
        if len(results) == 1 and isinstance(input, ParameterSet):
            return results[0]
        else:
            return results

    def _log_prob_single(self, sample: ParameterSet) -> float:
        """Log prior for a single instance under uniform bounds.

        Returns 0.0 if within bounds and satisfying constraints, otherwise
        ``-inf``.
        """
        if sample is None or not isinstance(sample, ParameterSet):
            raise ValueError("Sample must be an instance of ParameterSet.")
        result = 0.0
        for key, param in self.parameters.items():
            if isinstance(param, IndependentParameter) and param._is_sampled:
                if not (param.range[0] <= sample[key] <= param.range[1]):
                    result = -np.inf
                if not all(sample.satisfies(c) for c in self.constraints):
                    result = -np.inf
        return result

    def order(self, instance: ParameterSet) -> ParameterSet:
        """Reindex an instance to the bank's canonical parameter order.

        Args:
            instance: The ``ParameterSet`` to reindex.

        Returns:
            ParameterSet: A new instance with parameters ordered canonically.

        Raises:
            ValueError: If reindexing fails (e.g., missing keys).
        """
        if not isinstance(instance, ParameterSet):
            raise ValueError("Input must be an instance of ParameterSet.")
        try:
            out = instance.reindex(self.names)
        except Exception as e:
            raise ValueError("Error reordering parameters: " + str(e))
        return out

    def pretty_print(self) -> None:
        """Print a human-readable summary of the bank configuration."""
        print("ParameterBank:")
        print("----------------")
        for name, param in self.parameters.items():
            print(f"{name}: {param}")
        print("Constraints:")
        print("----------------")
        for constraint in self.constraints:
            print(constraint)
