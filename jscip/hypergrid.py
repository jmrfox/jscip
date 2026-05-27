"""HyperGrid class for jscip.

This module defines the HyperGrid class, which enables systematic evaluation
of parameter banks over a grid of values instead of random sampling.
"""

from __future__ import annotations

import logging
from itertools import product

import numpy as np

from .parameter_bank import ParameterBank
from .parameter_set import ParameterSet
from .parameters import IndependentScalarParameter

logger = logging.getLogger(__name__)


class HyperGrid:
    """Generate systematic grid evaluations over a ParameterBank.

    The HyperGrid creates a Cartesian product of sampled scalar parameters
    over their ranges using linear spacing, returning ParameterSet instances
    with derived parameters computed for each grid point.

    Args:
        parameter_bank: ParameterBank containing parameters to grid over.
        n_points: Number of grid points per parameter (must be positive integer).

    Raises:
        ValueError: If n_points is not positive, no sampled scalar parameters
            exist, or vector-valued parameters are marked for sampling.
    """

    def __init__(self, parameter_bank: ParameterBank, n_points: int) -> None:
        if not isinstance(parameter_bank, ParameterBank):
            raise ValueError(
                "parameter_bank must be an instance of ParameterBank."
            )

        if not isinstance(n_points, int) or n_points <= 0:
            raise ValueError("n_points must be a positive integer.")

        self.parameter_bank = parameter_bank
        self.n_points = n_points

        # Validate and extract sampled scalar parameters
        self._validate_parameters()
        self._extract_grid_parameters()

    def _validate_parameters(self) -> None:
        """Validate that the parameter bank is suitable for grid generation."""
        # Check if any sampled parameters are vector-valued
        vector_sampled = [
            name
            for name, param in self.parameter_bank.parameters.items()
            if param.is_sampled and hasattr(param, "shape")
        ]

        if vector_sampled:
            raise ValueError(
                f"Vector-valued parameters cannot be used in HyperGrid: "
                f"{vector_sampled}. Only scalar parameters are supported."
            )

        # Check if there are any sampled scalar parameters
        scalar_sampled = [
            name
            for name, param in self.parameter_bank.parameters.items()
            if param.is_sampled
            and isinstance(param, IndependentScalarParameter)
        ]

        if not scalar_sampled:
            raise ValueError(
                "No sampled scalar parameters found in ParameterBank. "
                "At least one scalar parameter must be marked as sampled."
            )

        # Check that all sampled scalar parameters have ranges
        missing_ranges = []
        for name, param in self.parameter_bank.parameters.items():
            if param.is_sampled and isinstance(
                param, IndependentScalarParameter
            ):
                if param.range is None:
                    missing_ranges.append(name)

        if missing_ranges:
            raise ValueError(
                f"Sampled scalar parameters must have ranges defined: {missing_ranges}"
            )

    def _extract_grid_parameters(self) -> None:
        """Extract and store information about parameters to grid over."""
        self.grid_params = {}

        for name, param in self.parameter_bank.parameters.items():
            if param.is_sampled and isinstance(
                param, IndependentScalarParameter
            ):
                assert param.range is not None  # Type guard for mypy
                self.grid_params[name] = {
                    "parameter": param,
                    "range": param.range,
                    "values": np.linspace(
                        param.range[0], param.range[1], self.n_points
                    ),
                }

        logger.debug(
            "Extracted %d grid parameters: %s",
            len(self.grid_params),
            list(self.grid_params.keys()),
        )

    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB for the generated grid."""
        # Estimate size of a single ParameterSet
        # Rough estimate: each parameter ~8 bytes (float) + overhead
        n_params = len(self.parameter_bank.parameters)
        bytes_per_set = n_params * 8 + 1000  # 1000 bytes overhead

        # Total number of grid points
        n_total_points = self.n_points ** len(self.grid_params)

        # Total memory in MB
        total_bytes = n_total_points * bytes_per_set
        total_mb = total_bytes / (1024 * 1024)

        return total_mb

    def generate(self) -> list[ParameterSet]:
        """Generate the grid of ParameterSet instances.

        Returns:
            List of ParameterSet instances covering the Cartesian product
            of all sampled scalar parameter ranges.

        Raises:
            RuntimeError: If grid generation fails unexpectedly.
        """
        # Check memory usage and warn if large
        estimated_mb = self._estimate_memory_usage()
        if estimated_mb > 100:  # Warning threshold
            logger.warning(
                f"Large grid detected! Estimated memory usage: "
                f"{estimated_mb:.1f} MB. "
                "This may consume significant memory."
            )

        # Generate grid points
        param_names = list(self.grid_params.keys())
        param_values = [
            self.grid_params[name]["values"] for name in param_names
        ]

        # Create Cartesian product
        grid_points = product(*param_values)
        n_total_points = self.n_points ** len(self.grid_params)

        logger.info(
            "Generating grid with %d points across %d parameters",
            n_total_points,
            len(self.grid_params),
        )

        # Generate ParameterSet instances
        results = []

        for point in grid_points:
            # Create parameter values for this grid point
            param_dict = {}

            # Set grid parameters
            for i, name in enumerate(param_names):
                param_dict[name] = point[i]

            # Set non-sampled independent parameters to their current values
            for name, param in self.parameter_bank.parameters.items():
                if not param.is_sampled and hasattr(param, "value"):
                    param_dict[name] = param.value

            # Create ParameterSet and compute derived parameters
            try:
                # Start with independent parameters
                base_set = ParameterSet(param_dict)

                # Compute derived parameters iteratively to handle dependencies
                # Start with base_set and keep computing derived parameters
                # until no new ones are found
                current_dict = dict(base_set)
                max_iterations = len(
                    self.parameter_bank.parameters
                )  # Prevent infinite loops

                for iteration in range(max_iterations):
                    new_derived = {}
                    for name, param in self.parameter_bank.parameters.items():
                        if (
                            hasattr(param, "compute")
                            and name not in current_dict
                        ):
                            try:
                                # Create temporary ParameterSet with current values
                                temp_set = ParameterSet(current_dict)
                                new_derived[name] = param.compute(temp_set)
                            except KeyError:
                                # Dependency not yet available, skip for now
                                continue

                    if not new_derived:
                        # No new derived parameters computed, we're done
                        break

                    current_dict.update(new_derived)

                # Create final ParameterSet
                final_set = ParameterSet(current_dict)

                # Order according to parameter bank
                ordered_set = self.parameter_bank.order(final_set)
                results.append(ordered_set)

            except Exception as e:
                raise RuntimeError(
                    f"Failed to generate grid point {point}: {e}"
                ) from e

        logger.info("Successfully generated %d grid points", len(results))
        return results

    def __repr__(self) -> str:
        return (
            f"HyperGrid(parameter_bank={self.parameter_bank}, "
            f"n_points={self.n_points}, "
            f"grid_parameters={list(self.grid_params.keys())})"
        )
