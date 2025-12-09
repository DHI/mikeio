"""Lazy, composable API for processing DFS files.

This module provides a Polars-inspired lazy evaluation API for efficient
processing of large DFS files without loading them entirely into memory.

Examples
--------
>>> from mikeio.lazy import scan_dfs
>>>
>>> # Simple pipeline
>>> (scan_dfs("large_file.dfsu")
...     .select(["Temperature", "Salinity"])
...     .filter(time=slice("2020-01-01", "2020-12-31"))
...     .rolling(window=24, stat="mean")
...     .to_dfs("output.dfsu")
... )
>>>
>>> # Custom transformations
>>> (scan_dfs("hourly.dfs0")
...     .select(["WaterLevel"])
...     .rolling(window=24, stat=lambda x: np.nanpercentile(x, 90))
...     .with_columns(WaterLevel=lambda x: x * 1.8 + 32)  # to Fahrenheit
...     .to_dfs("processed.dfs0")
... )

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from mikecore.DfsFile import DfsFile
from mikecore.DfsFileFactory import DfsFileFactory

# Import utilities from generic module
from .generic import _TimeInfo, _clone, _valid_item_numbers

__all__ = ["scan_dfs", "LazyDfs"]


# ============================================================================
# Operation Protocol and Base Classes
# ============================================================================


class Operation(ABC):
    """Base class for pipeline operations."""

    @abstractmethod
    def affects_metadata(self) -> bool:
        """Return True if this operation changes file structure (items, time axis)."""
        pass

    @abstractmethod
    def apply(self, context: ExecutionContext) -> None:
        """Apply the operation in the execution context."""
        pass


@dataclass
class ExecutionContext:
    """Execution context for pipeline operations.

    Holds state during pipeline execution including file handles,
    current metadata, and data buffers.
    """

    dfs_in: DfsFile
    dfs_out: DfsFile
    item_numbers: list[int]  # Which items from input to process
    time_info: _TimeInfo | None = None
    deletevalue: float = np.nan


# ============================================================================
# Concrete Operations
# ============================================================================


class SelectOp(Operation):
    """Operation to select specific items from the file."""

    def __init__(self, items: Sequence[int | str]):
        self.items = items

    def affects_metadata(self) -> bool:
        return True

    def apply(self, context: ExecutionContext) -> None:
        """Select items by updating the context's item_numbers."""
        # Use existing utility to validate and get item numbers
        context.item_numbers = _valid_item_numbers(context.dfs_in.ItemInfo, self.items)


class FilterOp(Operation):
    """Operation to filter timesteps."""

    def __init__(
        self,
        time: slice | None = None,
        start: int | float | str | datetime | None = None,
        end: int | float | str | datetime | None = None,
        step: int = 1,
    ):
        # Handle slice notation
        if time is not None:
            if not isinstance(time, slice):
                raise TypeError(f"time must be a slice, got {type(time)}")
            self.start = time.start if time.start is not None else 0
            self.end = time.stop if time.stop is not None else -1
            self.step = time.step if time.step is not None else 1
        else:
            self.start = start if start is not None else 0
            self.end = end if end is not None else -1
            self.step = step

    def affects_metadata(self) -> bool:
        return True

    def apply(self, context: ExecutionContext) -> None:
        """Parse time filter and store in context."""
        time_axis = context.dfs_in.FileInfo.TimeAxis
        context.time_info = _TimeInfo.parse(time_axis, self.start, self.end, self.step)


class WithColumnsOp(Operation):
    """Operation to transform items with custom functions."""

    def __init__(self, transforms: dict[str, Callable[[np.ndarray], np.ndarray]]):
        self.transforms = transforms

    def affects_metadata(self) -> bool:
        return False  # Same items, just transformed values

    def apply(self, context: ExecutionContext) -> None:
        """Store transforms for later application during data processing."""
        # This will be applied during the data processing loop
        # Store in context for the executor to use
        if not hasattr(context, "transforms"):
            context.transforms = {}  # type: ignore
        context.transforms.update(self.transforms)  # type: ignore


class RollingOp(Operation):
    """Operation to apply rolling window statistics."""

    def __init__(
        self,
        window: int,
        stat: str | Callable[[np.ndarray], float] = "mean",
        center: bool = False,
        min_periods: int | None = None,
    ):
        self.window = window
        self.stat = stat
        self.center = center
        self.min_periods = min_periods if min_periods is not None else window

        # Map string stats to numpy functions
        self._stat_functions = {
            "mean": np.nanmean,
            "min": np.nanmin,
            "max": np.nanmax,
            "median": np.nanmedian,
            "sum": np.nansum,
            "std": np.nanstd,
        }

    def affects_metadata(self) -> bool:
        return False  # Same structure, different values

    def get_stat_func(self) -> Callable[..., Any]:
        """Get the statistic function to apply."""
        if isinstance(self.stat, str):
            if self.stat not in self._stat_functions:
                raise ValueError(
                    f"Unknown stat '{self.stat}'. "
                    f"Available: {list(self._stat_functions.keys())}"
                )
            return self._stat_functions[self.stat]
        return self.stat

    def apply(self, context: ExecutionContext) -> None:
        """Store rolling config for later application during data processing."""
        # This will be applied during the data processing loop
        if not hasattr(context, "rolling"):
            context.rolling = None  # type: ignore
        context.rolling = self  # type: ignore


# ============================================================================
# LazyDfs - Main User-Facing Class
# ============================================================================


class LazyDfs:
    """Lazy DFS file processor with composable operations.

    This class provides a fluent interface for building pipelines of operations
    on DFS files. Operations are queued and executed lazily when .to_dfs() is called.

    Examples
    --------
    >>> df = scan_dfs("large_file.dfsu")
    >>> result = (df
    ...     .select(["Temperature", "Salinity"])
    ...     .filter(time=slice("2020-01-01", "2020-12-31"))
    ...     .rolling(window=24, stat="mean")
    ...     .to_dfs("output.dfsu")
    ... )

    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._operations: list[Operation] = []

    def select(self, items: Sequence[int | str]) -> LazyDfs:
        """Select specific items/variables.

        Parameters
        ----------
        items : Sequence[int | str]
            Items to select, either by index (0-based) or by name

        Returns
        -------
        LazyDfs
            Self for method chaining

        Examples
        --------
        >>> df.select(["Temperature", "Salinity"])
        >>> df.select([0, 1, 2])  # by index

        """
        self._operations.append(SelectOp(items))
        return self

    def filter(
        self,
        time: slice | None = None,
        start: int | float | str | datetime | None = None,
        end: int | float | str | datetime | None = None,
        step: int = 1,
    ) -> LazyDfs:
        """Filter timesteps.

        Parameters
        ----------
        time : slice, optional
            Time range as a slice (Polars-style). Either use this or start/end/step.
        start : int | float | str | datetime, optional
            Start of time range (step index, seconds, or datetime string)
        end : int | float | str | datetime, optional
            End of time range (step index, seconds, or datetime string)
        step : int, optional
            Step size (e.g., 2 for every other timestep), default 1

        Returns
        -------
        LazyDfs
            Self for method chaining

        Examples
        --------
        >>> df.filter(time=slice("2020-01-01", "2020-12-31"))
        >>> df.filter(time=slice(0, 100, 2))  # every other timestep
        >>> df.filter(start="2020-01-01", end="2020-12-31")

        """
        self._operations.append(FilterOp(time=time, start=start, end=end, step=step))
        return self

    def rolling(
        self,
        window: int,
        stat: str | Callable[[np.ndarray], float] = "mean",
        center: bool = False,
        min_periods: int | None = None,
    ) -> LazyDfs:
        """Apply rolling window statistic.

        Parameters
        ----------
        window : int
            Size of the rolling window (number of timesteps)
        stat : str | Callable, optional
            Either a built-in statistic name ('mean', 'min', 'max', 'median',
            'sum', 'std') or a custom function that takes a 1D array and returns
            a scalar. Default is 'mean'.
        center : bool, optional
            If True, set the window labels as the center of the window.
            If False (default), labels are the right edge of the window.
        min_periods : int, optional
            Minimum number of observations required to have a value.
            If None, defaults to window size.

        Returns
        -------
        LazyDfs
            Self for method chaining

        Examples
        --------
        >>> # 24-hour rolling mean
        >>> df.rolling(window=24, stat="mean")

        >>> # Rolling maximum with centered window
        >>> df.rolling(window=7, stat="max", center=True)

        >>> # Custom statistic
        >>> df.rolling(window=24, stat=lambda x: np.nanpercentile(x, 90))

        """
        self._operations.append(
            RollingOp(window=window, stat=stat, center=center, min_periods=min_periods)
        )
        return self

    def with_columns(self, **transforms: Callable[[np.ndarray], np.ndarray]) -> LazyDfs:
        """Transform items with custom functions.

        Parameters
        ----------
        **transforms
            Keyword arguments where keys are item names and values are
            transformation functions. Each function takes the item's data
            array and returns the transformed array.

        Returns
        -------
        LazyDfs
            Self for method chaining

        Examples
        --------
        >>> # Convert temperature from Celsius to Fahrenheit
        >>> df.with_columns(Temperature=lambda x: x * 1.8 + 32)

        >>> # Multiple transformations
        >>> df.with_columns(
        ...     Temperature=lambda x: x + 273.15,  # to Kelvin
        ...     Salinity=lambda x: x * 1.2,
        ... )

        """
        self._operations.append(WithColumnsOp(transforms))
        return self

    def to_dfs(self, path: str | Path, buffer_size: float = 1e9) -> None:
        """Execute pipeline and write to DFS file.

        This method triggers the actual execution of all queued operations.
        The file is processed efficiently in a single pass.

        Parameters
        ----------
        path : str | Path
            Output file path
        buffer_size : float, optional
            Maximum memory buffer size in bytes for processing large files,
            default 1e9 (1 GB)

        Examples
        --------
        >>> (scan_dfs("input.dfsu")
        ...     .select(["Temperature"])
        ...     .rolling(window=24, stat="mean")
        ...     .to_dfs("output.dfsu")
        ... )

        """
        executor = PipelineExecutor(
            infilename=self.path,
            outfilename=Path(path),
            operations=self._operations,
            buffer_size=buffer_size,
        )
        executor.execute()


# ============================================================================
# Pipeline Executor
# ============================================================================


class PipelineExecutor:
    """Executes a pipeline of operations on a DFS file."""

    def __init__(
        self,
        infilename: Path,
        outfilename: Path,
        operations: list[Operation],
        buffer_size: float = 1e9,
    ):
        self.infilename = infilename
        self.outfilename = outfilename
        self.operations = operations
        self.buffer_size = buffer_size

    def execute(self) -> None:
        """Execute the pipeline."""
        # Open input file
        dfs_in = DfsFileFactory.DfsGenericOpen(str(self.infilename))

        try:
            # Initialize execution context
            context = ExecutionContext(
                dfs_in=dfs_in,
                dfs_out=None,  # type: ignore # Will be set after metadata ops
                item_numbers=list(range(len(dfs_in.ItemInfo))),  # All items initially
                deletevalue=dfs_in.FileInfo.DeleteValueFloat,
            )

            # Apply metadata operations to determine what to process
            self._apply_metadata_operations(context)

            # Apply data operations to prepare context
            self._apply_data_operations(context)

            # Create output file with appropriate metadata
            dfs_out = self._create_output_file(context)
            context.dfs_out = dfs_out

            # Process data
            self._process_data(context)

            # Close files
            dfs_out.Close()

        finally:
            dfs_in.Close()

    def _apply_metadata_operations(self, context: ExecutionContext) -> None:
        """Apply operations that affect metadata (select, filter)."""
        for op in self.operations:
            if op.affects_metadata():
                op.apply(context)

    def _apply_data_operations(self, context: ExecutionContext) -> None:
        """Apply operations that transform data (with_columns, rolling)."""
        for op in self.operations:
            if not op.affects_metadata():
                op.apply(context)

    def _create_output_file(self, context: ExecutionContext) -> DfsFile:
        """Create output file with appropriate metadata."""
        # Determine output time axis parameters
        start_time = None
        timestep = None
        if context.time_info is not None:
            start_time = context.time_info.file_start_new
            timestep = context.time_info.timestep

        # Clone file structure with selected items
        dfs_out = _clone(
            infilename=str(self.infilename),
            outfilename=str(self.outfilename),
            start_time=start_time,
            timestep=timestep,
            items=context.item_numbers,
        )

        return dfs_out

    def _process_data(self, context: ExecutionContext) -> None:
        """Process and write data through the pipeline."""
        # Check for rolling operation
        rolling_op = getattr(context, "rolling", None)

        if rolling_op is not None:
            self._process_with_rolling(context, rolling_op)
        else:
            self._process_simple(context)

    def _process_simple(self, context: ExecutionContext) -> None:
        """Process data without rolling (simple passthrough with transforms)."""
        time_info = context.time_info

        # Determine time range
        if time_info is not None:
            start_step = time_info.start_step
            end_step = time_info.end_step
            step = time_info.timestep if time_info.timestep else 1
            start_sec = time_info.start_sec
            end_sec = time_info.end_sec
        else:
            start_step = 0
            end_step = context.dfs_in.FileInfo.TimeAxis.NumberOfTimeSteps
            step = 1
            start_sec = -np.inf
            end_sec = np.inf

        # Get transforms if any
        transforms = getattr(context, "transforms", {})

        # Get item names for transforms
        item_names = [context.dfs_in.ItemInfo[i].Name for i in context.item_numbers]

        timestep_out = -1
        for timestep in range(start_step, end_step, int(step)):
            for item_idx, item_num in enumerate(context.item_numbers):
                itemdata = context.dfs_in.ReadItemTimeStep(item_num + 1, timestep)
                time_sec = itemdata.Time

                # Check time bounds
                if time_sec > end_sec:
                    return
                if time_sec < start_sec:
                    continue

                # Track output timestep
                if item_idx == 0:
                    timestep_out += 1

                # Get data
                data = itemdata.Data

                # Apply transforms if any
                item_name = item_names[item_idx]
                if item_name in transforms:
                    # Convert delete value to NaN before transform
                    mask = data == context.deletevalue
                    data[mask] = np.nan

                    # Apply transform
                    data = transforms[item_name](data)

                    # Convert NaN back to delete value
                    data[np.isnan(data)] = context.deletevalue

                # Write to output
                context.dfs_out.WriteItemTimeStep(
                    item_idx + 1, timestep_out, time_sec, data
                )

    def _process_with_rolling(
        self, context: ExecutionContext, rolling_op: RollingOp
    ) -> None:
        """Process data with rolling window operation."""
        window = rolling_op.window
        stat_func = rolling_op.get_stat_func()
        min_periods = rolling_op.min_periods
        center = rolling_op.center

        n_items = len(context.item_numbers)
        time_info = context.time_info

        # Determine time range
        if time_info is not None:
            start_step = time_info.start_step
            end_step = time_info.end_step
        else:
            start_step = 0
            end_step = context.dfs_in.FileInfo.TimeAxis.NumberOfTimeSteps

        # Get data shape from first item
        first_itemdata = context.dfs_in.ReadItemTimeStep(
            context.item_numbers[0] + 1, start_step
        )
        n_elements = len(first_itemdata.Data)

        # Create buffers: [n_items, window, n_elements]
        buffers = [
            np.full((window, n_elements), np.nan, dtype=np.float32)
            for _ in range(n_items)
        ]
        buffer_idx = 0  # Current position in circular buffer

        # Process timesteps
        for timestep_in in range(start_step, end_step):
            # Read all items for this timestep
            for item_idx, item_num in enumerate(context.item_numbers):
                itemdata = context.dfs_in.ReadItemTimeStep(item_num + 1, timestep_in)
                data = itemdata.Data

                # Replace delete values with NaN
                data = data.astype(np.float32)
                data[data == context.deletevalue] = np.nan

                # Store in buffer (circular)
                buffers[item_idx][buffer_idx % window] = data

            # Calculate how many valid timesteps we have
            timesteps_filled = min(timestep_in - start_step + 1, window)

            # Only output if we have enough periods
            if timesteps_filled >= min_periods:
                # Calculate output timestep
                if center:
                    # For centered window, output is delayed
                    timestep_out = timestep_in - start_step - window // 2
                    if timestep_out < 0:
                        buffer_idx += 1
                        continue
                else:
                    timestep_out = timestep_in - start_step

                # Compute rolling statistic for each item
                for item_idx in range(n_items):
                    # Get the window data
                    if timesteps_filled < window:
                        # Partial window at the start
                        window_data = buffers[item_idx][:timesteps_filled]
                    else:
                        # Full window - need to handle circular buffer
                        window_data = buffers[item_idx]

                    # Apply statistic along time axis (axis=0)
                    result = np.apply_along_axis(stat_func, 0, window_data)

                    # Replace NaN with delete value
                    result[np.isnan(result)] = context.deletevalue

                    # Write to output (use WriteItemTimeStepNext for sequential writing)
                    itemdata = context.dfs_in.ReadItemTimeStep(
                        context.item_numbers[item_idx] + 1, timestep_in
                    )
                    context.dfs_out.WriteItemTimeStepNext(itemdata.Time, result)

            buffer_idx += 1


# ============================================================================
# Factory Function
# ============================================================================


def scan_dfs(path: str | Path) -> LazyDfs:
    """Create a lazy DFS file processor.

    This is the entry point for the lazy evaluation API. Operations are
    queued and executed when .to_dfs() is called.

    Parameters
    ----------
    path : str | Path
        Path to the input DFS file

    Returns
    -------
    LazyDfs
        Lazy processor for building operation pipelines

    Examples
    --------
    >>> from mikeio.lazy import scan_dfs
    >>>
    >>> df = scan_dfs("large_file.dfsu")
    >>> result = (df
    ...     .select(["Temperature", "Salinity"])
    ...     .filter(time=slice("2020-01-01", "2020-12-31"))
    ...     .rolling(window=24, stat="mean")
    ...     .to_dfs("output.dfsu")
    ... )

    """
    return LazyDfs(path)
