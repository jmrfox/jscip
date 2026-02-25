"""ParameterBank class for jscip.

This module defines the ParameterBank class, which manages collections of
parameters with sampling, constraints, and conversions.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator

import numpy as np
import pandas as pd

from .parameter_set import ParameterSet
from .parameters import (
    DerivedScalarParameter,
    DerivedVectorParameter,
    IndependentScalarParameter,
    IndependentVectorParameter,
)

logger = logging.getLogger(__name__)


class ParameterBank:
    """A collection of parameters with sampling, constraints, and conversions.

    The bank stores independent and derived parameters, optional constraint
    functions, and a canonical parameter order. It can sample full parameter
    instances, validate them against constraints, and convert between rich
    ``ParameterSet`` and array/dataframe representations.

    Args:
        parameters: Dictionary mapping parameter names to parameter instances.
        constraints: List of constraint functions that take a ParameterSet and
            return a boolean.
        array_mode: If True, sampling and conversions use only sampled
            parameters and return plain arrays; otherwise use all parameters
            and return ParameterSet objects.
        texnames: Optional dictionary mapping parameter names to TeX-formatted
            display names.
        max_attempts: Maximum number of attempts when sampling with constraints
            before raising an error. Defaults to 100.
    """

    def __init__(
        self,
        parameters: (
            dict[
                str,
                IndependentScalarParameter | IndependentVectorParameter | DerivedScalarParameter,
            ]
            | None
        ) = None,
        constraints: list[Callable[[ParameterSet], bool]] | None = None,
        array_mode: bool = False,
        texnames: dict[str, str] | None = None,
        max_attempts: int = 100,
    ) -> None:
        self.parameters = parameters if parameters is not None else {}
        self.constraints = constraints if constraints is not None else []
        self.array_mode = array_mode
        if not isinstance(max_attempts, int) or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer.")
        self._max_attempts = max_attempts
        self.texnames = texnames if texnames is not None else {}

        for key, value in self.parameters.items():
            if not isinstance(
                value,
                (
                    IndependentScalarParameter,
                    IndependentVectorParameter,
                    DerivedScalarParameter,
                    DerivedVectorParameter,
                ),
            ):
                raise ValueError(
                    f"Value for key '{key}' must be an instance of "
                    f"IndependentScalarParameter, IndependentVectorParameter, "
                    f"DerivedScalarParameter, or DerivedVectorParameter."
                )

        # Validate texnames keys match parameter names
        if self.texnames:
            invalid_keys = set(self.texnames.keys()) - set(self.parameters.keys())
            if invalid_keys:
                raise ValueError(
                    f"texnames contains keys not in parameters: {invalid_keys}. "
                    f"Valid parameter names are: {set(self.parameters.keys())}"
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
            if isinstance(param, (IndependentScalarParameter, IndependentVectorParameter))
            and param.is_sampled
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
        return [key for key, param in self.parameters.items() if param.is_sampled]

    @property
    def vector_names(self) -> list[str]:
        """Get a list of all vector parameter names."""
        return [
            key
            for key, param in self.parameters.items()
            if isinstance(param, IndependentVectorParameter)
        ]

    @property
    def lower_bounds(self) -> np.ndarray:
        """Get the lower bounds of all sampled parameters.

        For scalar parameters, returns the scalar lower bound.
        For vector parameters, returns the lower bound array or scalar if uniform.
        """
        bounds = []
        for _key, param in self.parameters.items():
            if isinstance(param, IndependentScalarParameter) and param.is_sampled:
                bounds.append(param.range[0])
            elif isinstance(param, IndependentVectorParameter) and param.is_sampled:
                # Vector parameter range can be tuple of arrays or scalars
                if isinstance(param.range[0], np.ndarray):
                    bounds.extend(param.range[0])
                else:
                    bounds.extend([param.range[0]] * param.shape[0])
        return np.array(bounds)

    @property
    def upper_bounds(self) -> np.ndarray:
        """Get the upper bounds of all sampled parameters.

        For scalar parameters, returns the scalar upper bound.
        For vector parameters, returns the upper bound array or scalar if uniform.
        """
        bounds = []
        for _key, param in self.parameters.items():
            if isinstance(param, IndependentScalarParameter) and param.is_sampled:
                bounds.append(param.range[1])
            elif isinstance(param, IndependentVectorParameter) and param.is_sampled:
                # Vector parameter range can be tuple of arrays or scalars
                if isinstance(param.range[1], np.ndarray):
                    bounds.extend(param.range[1])
                else:
                    bounds.extend([param.range[1]] * param.shape[0])
        return np.array(bounds)

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

    def __getitem__(
        self, key: str
    ) -> IndependentScalarParameter | IndependentVectorParameter | DerivedScalarParameter:
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
            array_mode=self.array_mode,
            texnames=self.texnames.copy(),
            max_attempts=self._max_attempts,
        )
        logger.debug("Copied ParameterBank: %s", result)
        return result

    def get_value(self, key: str) -> float:
        if key in self.parameters:
            return self.parameters[key].value
        else:
            raise KeyError(f"Parameter '{key}' not found in the bank.")

    def merge(self, other: ParameterBank) -> None:
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
        self,
        name: str,
        parameter: (
            IndependentScalarParameter
            | IndependentVectorParameter
            | DerivedScalarParameter
            | DerivedVectorParameter
        ),
    ) -> None:
        """Add a new parameter to the bank."""
        if not isinstance(
            parameter,
            (
                IndependentScalarParameter,
                IndependentVectorParameter,
                DerivedScalarParameter,
            ),
        ):
            raise ValueError(
                "Parameter must be an instance of IndependentScalarParameter, "
                "IndependentVectorParameter, or DerivedScalarParameter."
            )
        if name in self.parameters:
            raise KeyError(f"Parameter '{name}' already exists in the bank.")
        self.parameters[name] = parameter
        self._refresh_sampled_indices()
        logger.debug("Added parameter '%s' to ParameterBank: %s", name, self)

    def add_constraint(self, constraint: Callable[[ParameterSet], bool]) -> None:
        """Add a new constraint to the bank."""
        if not callable(constraint):
            raise ValueError("Constraint must be a callable function.")
        self.constraints.append(constraint)
        logger.debug("Added constraint '%s' to ParameterBank: %s", constraint, self)

    def get_constraints(self) -> list[Callable[[ParameterSet], bool]]:
        """Get all constraints in the bank."""
        return self.constraints

    def get_default_values(self, return_array: bool | None = None) -> ParameterSet | np.ndarray:
        """Return default values for all parameters.

        Computes a ``ParameterSet`` by taking the current ``value`` for all
        independent parameters and computing all derived parameters from those
        values. Optionally, returns the sampled subset as a NumPy array when
        ``return_array=True``.

        Args:
            return_array: If True, return a 1D NumPy array of sampled parameter
                values in canonical sampled order. If False, return a full
                ``ParameterSet``. Defaults to ``self.array_mode``.

        Returns:
            ParameterSet | numpy.ndarray: The default instance or the sampled
            values array.

        Raises:
            ValueError: If ``return_array`` is not a boolean.
        """
        if return_array is None:
            return_array = self.array_mode
        if not isinstance(return_array, bool):
            raise ValueError("return_array must be a boolean value.")
        p = ParameterSet(
            {
                key: param.value
                for key, param in self.parameters.items()
                if isinstance(param, (IndependentScalarParameter, IndependentVectorParameter))
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
                    if isinstance(param, DerivedScalarParameter)
                },
            }
        )
        p = self.order(p)
        logger.debug("[get_default_values] Default values for all parameters in the bank: %s", p)
        if return_array:
            return self.instance_to_array(p)
        else:
            return p

    def instance_to_array(self, input: ParameterSet | list[ParameterSet]) -> np.ndarray:
        """Convert a parameter instance (or list) to a sampled parameter array.

        Args:
            input: A single ``ParameterSet`` or list of ``ParameterSet``
                instances.

        Returns:
            numpy.ndarray: 1D array for a single instance or 2D array for a
            list of instances, containing values for sampled parameters only,
            in canonical sampled order. Vector parameters are flattened into
            the array.

        Raises:
            ValueError: If ``input`` is not a ``ParameterSet`` or list thereof.
        """
        if not isinstance(input, (ParameterSet, list)):
            raise ValueError(
                "Input must be a ParameterSetInstance or a list of ParameterSetInstances."
            )
        if isinstance(input, ParameterSet):
            # Flatten vector parameters into the theta array
            theta_values = []
            for key in self.sampled:
                value = input[key]
                if isinstance(value, np.ndarray):
                    theta_values.extend(value)
                else:
                    theta_values.append(value)
            theta = np.array(theta_values)
            logger.debug("[instance_to_array] Converted ParameterSet to numpy array: %s", theta)
        else:
            # return a 2D array of shape (n_instances, n_theta_dims)
            theta_list = []
            for instance in input:
                theta_values = []
                for key in self.sampled:
                    value = instance[key]
                    if isinstance(value, np.ndarray):
                        theta_values.extend(value)
                    else:
                        theta_values.append(value)
                theta_list.append(theta_values)
            theta = np.array(theta_list)
            logger.debug(
                "[instance_to_array] Converted list of ParameterSetInstances to numpy array: %s",
                theta,
            )
        return theta

    def dataframe_to_array(self, df: pd.DataFrame) -> np.ndarray:
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

    def array_to_instance(self, theta: np.ndarray) -> ParameterSet:
        """Convert a parameter array to a parameter instance.

        When ``array_mode`` is True, ``theta`` must contain only sampled
        independent parameters in canonical sampled order. Otherwise, it must
        contain values for all independent parameters in canonical order.

        Args:
            theta: 1D NumPy array.

        Returns:
            ParameterSet: A full instance with derived parameters recomputed.

        Raises:
            ValueError: If shapes are inconsistent with ``array_mode`` or
                if ``theta`` is not a NumPy array.
        """
        if not isinstance(theta, np.ndarray):
            raise ValueError("Input must be a numpy array, instead got: " + str(type(theta)))
        # validate length depending on array_mode
        if self.array_mode:
            # Calculate expected theta length (accounting for vector parameters)
            expected_len = len(self.lower_bounds)  # This already accounts for vectors
            if len(theta) != expected_len:
                raise ValueError(
                    f"Array length {len(theta)} does not match expected theta dimensions {expected_len}."
                )
        else:
            if len(theta) != len(self.parameters):
                raise ValueError(
                    f"Array length {len(theta)} does not match number of parameters {len(self.parameters)}."
                )
        # theta in this case must be a 1D array
        # Start with defaults
        out = self.get_default_values(return_array=False)
        if self.array_mode:
            # theta provides only sampled independent parameters
            # Need to unflatten vector parameters
            theta_idx = 0
            for key in self.sampled:
                param = self.parameters[key]
                if isinstance(param, IndependentVectorParameter):
                    # Extract vector elements from theta
                    n_elements = param.shape[0]
                    out[key] = theta[theta_idx : theta_idx + n_elements]
                    theta_idx += n_elements
                else:
                    # Scalar parameter
                    out[key] = theta[theta_idx]
                    theta_idx += 1
        else:
            # theta provides values for ALL parameters in canonical order
            if len(theta) != len(self.parameters):
                raise ValueError(
                    f"Array length {len(theta)} does not match number of parameters {len(self.parameters)}."
                )
            for i, key in enumerate(self.names):
                param = self.parameters[key]
                if isinstance(param, IndependentScalarParameter):
                    out[key] = float(theta[i])
        # recompute derived parameters
        out = ParameterSet(
            {
                **out,
                **{
                    key: param.compute(out)
                    for key, param in self.parameters.items()
                    if isinstance(param, DerivedScalarParameter)
                },
            }
        )
        return out

    def _sample_once(self) -> ParameterSet:
        """Sample a single full parameter set (internal).

        Samples all sampled independent parameters (scalar and vector), computes
        derived values, and returns a ``ParameterSet`` ordered canonically.
        """
        # first, sample all independent parameters (both sampled and fixed)
        p = ParameterSet(
            {
                key: param.sample()
                for key, param in self.parameters.items()
                if isinstance(param, (IndependentScalarParameter, IndependentVectorParameter))
            }
        )
        logger.debug(
            "[sample_once] Sampled values for all independent parameters in the bank: %s",
            p,
        )
        # then, compute all derived parameters based on the sampled independent parameters
        for key, param in self.parameters.items():
            if isinstance(param, (DerivedScalarParameter, DerivedVectorParameter)):
                p[key] = param.compute(p)
        logger.debug("[sample_once] Sampled values for all parameters in the bank: %s", p)
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
                only supported when ``array_mode`` is True.

        Returns:
            ParameterSet | pandas.DataFrame | numpy.ndarray: Depending on
            ``array_mode`` and ``size``.

        Raises:
            ValueError: If ``size`` has an invalid type or dimensionality.
        """
        if size is not None and not isinstance(size, int) and not isinstance(size, tuple):
            raise ValueError("Size must be None, an integer, or a tuple.")
        if size is None:
            n_samples = 1
        elif isinstance(size, int):
            n_samples = size
        elif isinstance(size, tuple):
            if len(size) > 1 and not self.array_mode:
                raise ValueError("Multiple dimensions are only supported for array_mode.")
            if len(size) == 1:
                n_samples = size[0]
            else:
                n_samples = int(np.prod(size))

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
                        if isinstance(
                            param,
                            (IndependentScalarParameter, IndependentVectorParameter),
                        )
                        and not param.is_sampled
                    },
                }
            )
            samples.append(sample)
        if self.array_mode:
            if size is None:
                out = self.instance_to_array(samples[0])
            elif isinstance(size, int):
                array_dim = len(self.lower_bounds)  # Accounts for vector parameters
                out = np.array([self.instance_to_array(sample) for sample in samples]).reshape(
                    (size, array_dim)
                )
            elif isinstance(size, tuple):
                array_dim = len(self.lower_bounds)  # Accounts for vector parameters
                out = np.array([self.instance_to_array(sample) for sample in samples]).reshape(
                    size + (array_dim,)
                )
        else:
            if size is None:
                out = samples[0]
            elif isinstance(size, int):
                out = self.instances_to_dataframe(list(samples))
            elif isinstance(size, tuple):
                out = self.instances_to_dataframe(list(samples))
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
            raise ValueError("Instances must be a list of ParameterSetInstance objects.")
        if not instances:
            raise ValueError("Instances list cannot be empty.")
        if not all(isinstance(instance, ParameterSet) for instance in instances):
            raise ValueError("All items in instances must be ParameterSetInstance objects.")
        df = pd.DataFrame([dict(instance) for instance in instances])
        # Don't convert to float if we have vector parameters (arrays)
        # Only convert scalar columns to float
        for col in df.columns:
            if col in self.vector_names:
                # Keep as object dtype for vector parameters
                continue
            else:
                df[col] = df[col].astype(float)
        df = df[self.names]  # reorder columns to canonical order
        return df

    def log_prob(self, input: ParameterSet | pd.DataFrame | np.ndarray) -> float | np.ndarray:
        """Compute log-probability for parameter instances.

        Args:
            input: A ``ParameterSet``, a pandas ``DataFrame`` (rows are
                instances), or a NumPy array (arrays, the
                expected width depends on ``theta_sampling``.

        Returns:
            float | numpy.ndarray: A scalar for a single ``ParameterSet`` or a
            NumPy array of log-probabilities for batches.

        Raises:
            ValueError: If the type/shape of ``input`` is inconsistent with the
                current ``array_mode`` mode.
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
                    input.shape[0] != len(self.sampled) and self.array_mode
                ):  # if array_mode is enabled, sample must match sampled parameters
                    raise ValueError(
                        f"1D numpy array must have length {len(self.sampled)} to match sampled parameters, since array_mode is enabled."
                    )
                elif (
                    input.shape[0] != len(self.parameters) and not self.array_mode
                ):  # if array_mode is disabled, sample must match all parameters
                    raise ValueError(
                        f"1D numpy array must have length {len(self.parameters)} to match all parameters, since array_mode is disabled."
                    )
                samples = [self.array_to_instance(input)]
            elif input.ndim == 2:  # if 2D, treat each row as a sample
                if input.shape[1] != len(self.sampled) and self.array_mode:
                    raise ValueError(
                        f"2D numpy array must have {len(self.sampled)} columns to match sampled parameters, since array_mode is enabled."
                    )
                elif input.shape[1] != len(self.parameters) and not self.array_mode:
                    raise ValueError(
                        f"2D numpy array must have {len(self.parameters)} columns to match all parameters, since array_mode is disabled."
                    )
                samples = [self.array_to_instance(row) for row in input]
            else:
                raise ValueError("Samples must be a 1D or 2D numpy array.")
        elif not isinstance(input, list):
            raise ValueError("Samples must be a list of ParameterSet instances or a numpy array.")

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
            if isinstance(param, IndependentScalarParameter) and param.is_sampled:
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
            raise ValueError("Error reordering parameters: " + str(e)) from e
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
