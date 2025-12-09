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
...     .with_items(WaterLevel=lambda x: x * 1.8 + 32)  # to Fahrenheit
...     .to_dfs("processed.dfs0")
... )

"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

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

    @property
    @abstractmethod
    def affects_metadata(self) -> bool:
        """Return True if this operation changes file structure (items, time axis)."""
        pass

    @abstractmethod
    def apply(self, context: ExecutionContext) -> None:
        """Apply the operation in the execution context."""
        pass

    @abstractmethod
    def name(self) -> str:
        """Return the name of this operation."""
        pass

    @abstractmethod
    def explain(self) -> str:
        """Return a detailed explanation of what this operation does."""
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
    time_step: int = 1  # Step size for iteration (e.g., 2 for every other timestep)
    deletevalue: float = np.nan
    # Operation-specific state
    transforms: dict[str, Callable[[np.ndarray], np.ndarray]] | None = None
    scale_params: tuple[float, float] | None = None
    rolling: "RollingOp | None" = None
    aggregate: "AggregateOp | None" = None
    diff_file: Path | None = None


# ============================================================================
# Concrete Operations
# ============================================================================


class SelectOp(Operation):
    """Operation to select specific items from the file."""

    def __init__(self, items: Sequence[int | str]):
        self.items = items

    @property
    def affects_metadata(self) -> bool:
        return True

    def name(self) -> str:
        return "select"

    def explain(self) -> str:
        items_str = ", ".join(str(item) for item in self.items)
        return f"SELECT items: [{items_str}]"

    def apply(self, context: ExecutionContext) -> None:
        """Select items by updating the context's item_numbers."""
        # Check if this is a layered dfsu (3d mesh with Z coordinate)
        is_layered = context.dfs_in.ItemInfo[0].Name == "Z coordinate"

        # Use existing utility to validate and get item numbers
        # For layered dfsu, ignore the first item (Z coordinate) during selection
        item_numbers = _valid_item_numbers(
            context.dfs_in.ItemInfo, self.items, ignore_first=is_layered
        )

        # For layered dfsu, shift item numbers and add Z coordinate at position 0
        if is_layered:
            item_numbers = [it + 1 for it in item_numbers]
            item_numbers.insert(0, 0)

        context.item_numbers = item_numbers


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

    @property
    def affects_metadata(self) -> bool:
        return True

    def name(self) -> str:
        return "filter"

    def explain(self) -> str:
        parts = []
        if self.start != 0:
            parts.append(f"start={self.start}")
        if self.end != -1:
            parts.append(f"end={self.end}")
        if self.step != 1:
            parts.append(f"step={self.step}")
        filter_str = ", ".join(parts) if parts else "all timesteps"
        return f"FILTER time: {filter_str}"

    def apply(self, context: ExecutionContext) -> None:
        """Parse time filter and store in context."""
        time_axis = context.dfs_in.FileInfo.TimeAxis
        context.time_info = _TimeInfo.parse(time_axis, self.start, self.end, self.step)
        context.time_step = self.step  # Store step for iteration


class WithItemsOp(Operation):
    """Operation to transform items with custom functions."""

    def __init__(self, transforms: dict[str, Callable[[np.ndarray], np.ndarray]]):
        self.transforms = transforms

    @property
    def affects_metadata(self) -> bool:
        return False  # Same items, just transformed values

    def name(self) -> str:
        return "with_items"

    def explain(self) -> str:
        items = ", ".join(self.transforms.keys())
        return f"TRANSFORM items: {items}"

    def apply(self, context: ExecutionContext) -> None:
        """Store transforms for later application during data processing."""
        # This will be applied during the data processing loop
        # Store in context for the executor to use
        if context.transforms is None:
            context.transforms = {}
        context.transforms.update(self.transforms)


class ScaleOp(Operation):
    """Operation to scale all items with factor and offset."""

    def __init__(self, factor: float = 1.0, offset: float = 0.0):
        self.factor = factor
        self.offset = offset

    @property
    def affects_metadata(self) -> bool:
        return False  # Same items, just scaled values

    def name(self) -> str:
        return "scale"

    def explain(self) -> str:
        return f"SCALE: data × {self.factor} + {self.offset}"

    def apply(self, context: ExecutionContext) -> None:
        """Store scale params for later application during data processing."""
        context.scale_params = (self.factor, self.offset)


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

        # Validate stat function
        if isinstance(stat, str) and stat not in self._stat_functions:
            raise ValueError(
                f"Unknown stat '{stat}'. Available: {list(self._stat_functions.keys())}"
            )
        self.stat = stat

    @property
    def affects_metadata(self) -> bool:
        return False  # Same structure, different values

    def name(self) -> str:
        return "rolling"

    def explain(self) -> str:
        parts = [f"window={self.window}", f"stat={self.stat}"]
        if self.center:
            parts.append("center=True")
        if self.min_periods != self.window:
            parts.append(f"min_periods={self.min_periods}")
        return f"ROLLING: {', '.join(parts)}"

    def get_stat_func(self) -> Callable[..., Any]:
        """Get the statistic function to apply."""
        if isinstance(self.stat, str):
            if self.stat not in self._stat_functions:
                raise ValueError(
                    f"Unknown stat '{self.stat}'. "
                    f"Available: {list(self._stat_functions.keys())}"
                )
            return cast(Callable[..., Any], self._stat_functions[self.stat])
        return cast(Callable[..., Any], self.stat)

    def apply(self, context: ExecutionContext) -> None:
        """Store rolling config for later application during data processing."""
        # This will be applied during the data processing loop
        context.rolling = self


class AggregateOp(Operation):
    """Operation to aggregate over time dimension."""

    def __init__(self, stat: str | Callable[[np.ndarray], np.ndarray]):
        # Map string stats to numpy functions
        self._stat_functions = {
            "mean": np.nanmean,
            "min": np.nanmin,
            "max": np.nanmax,
            "median": np.nanmedian,
            "sum": np.nansum,
            "std": np.nanstd,
        }

        # Validate stat function
        if isinstance(stat, str) and stat not in self._stat_functions:
            raise ValueError(
                f"Unknown stat '{stat}'. Available: {list(self._stat_functions.keys())}"
            )
        self.stat = stat

    @property
    def affects_metadata(self) -> bool:
        return True  # Output has single timestep

    def name(self) -> str:
        return "aggregate"

    def explain(self) -> str:
        stat = self.stat if isinstance(self.stat, str) else "<custom function>"
        return f"AGGREGATE: stat={stat} (→ single timestep)"

    def get_stat_func(self) -> Callable[..., Any]:
        """Get the statistic function to apply."""
        if isinstance(self.stat, str):
            if self.stat not in self._stat_functions:
                raise ValueError(
                    f"Unknown stat '{self.stat}'. "
                    f"Available: {list(self._stat_functions.keys())}"
                )
            return cast(Callable[..., Any], self._stat_functions[self.stat])
        return cast(Callable[..., Any], self.stat)

    def apply(self, context: ExecutionContext) -> None:
        """Store aggregate config for later application."""
        context.aggregate = self


class DiffOp(Operation):
    """Operation to compute difference with another file."""

    def __init__(self, other_file: str | Path):
        self.other_file = Path(other_file)

    @property
    def affects_metadata(self) -> bool:
        return False  # Same structure, different values

    def name(self) -> str:
        return "diff"

    def explain(self) -> str:
        return f"DIFF: subtract {self.other_file.name}"

    def apply(self, context: ExecutionContext) -> None:
        """Store diff config for later application."""
        context.diff_file = self.other_file


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

    def __repr__(self) -> str:
        """Return string representation of the lazy pipeline."""
        ops_str = " → ".join(op.name() for op in self._operations)
        if ops_str:
            return f"<LazyDfs: {self.path.name} | {ops_str}>"
        return f"<LazyDfs: {self.path.name}>"

    def _repr_html_(self) -> str:
        """Return HTML representation for Jupyter notebooks."""
        ops_html = []
        for i, op in enumerate(self._operations):
            ops_html.append(
                f'<div class="lazy-op">'
                f'<span class="op-number">{i+1}</span>'
                f'<span class="op-name">{op.name()}</span>'
                f'<div class="op-details">{op.explain()}</div>'
                f"</div>"
            )

        ops_section = (
            "\n".join(ops_html)
            if ops_html
            else '<div class="no-ops">No operations (will copy all data)</div>'
        )

        return f"""
        <style>
        .lazy-pipeline {{
            font-family: monospace;
            border: 2px solid #4CAF50;
            border-radius: 8px;
            padding: 15px;
            background: #f5f5f5;
            max-width: 600px;
        }}
        .pipeline-header {{
            font-size: 16px;
            font-weight: bold;
            color: #4CAF50;
            margin-bottom: 10px;
        }}
        .pipeline-input {{
            background: #e8f5e9;
            padding: 8px;
            border-radius: 4px;
            margin-bottom: 15px;
        }}
        .lazy-op {{
            background: white;
            margin: 8px 0;
            padding: 10px;
            border-left: 4px solid #2196F3;
            border-radius: 4px;
            display: flex;
            align-items: baseline;
            gap: 10px;
        }}
        .op-number {{
            background: #2196F3;
            color: white;
            padding: 2px 8px;
            border-radius: 50%;
            font-weight: bold;
            font-size: 12px;
        }}
        .op-name {{
            font-weight: bold;
            color: #1976D2;
        }}
        .op-details {{
            color: #666;
            flex: 1;
        }}
        .no-ops {{
            color: #999;
            font-style: italic;
            padding: 10px;
        }}
        </style>
        <div class="lazy-pipeline">
            <div class="pipeline-header">🔄 Lazy DFS Pipeline</div>
            <div class="pipeline-input">📁 Input: {self.path}</div>
            <div class="pipeline-ops">
                {ops_section}
            </div>
        </div>
        """

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

    def with_items(self, **transforms: Callable[[np.ndarray], np.ndarray]) -> LazyDfs:
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
        >>> df.with_items(Temperature=lambda x: x * 1.8 + 32)

        >>> # Multiple transformations
        >>> df.with_items(
        ...     Temperature=lambda x: x + 273.15,  # to Kelvin
        ...     Salinity=lambda x: x * 1.2,
        ... )

        """
        self._operations.append(WithItemsOp(transforms))
        return self

    def scale(self, factor: float = 1.0, offset: float = 0.0) -> LazyDfs:
        """Scale all items with a factor and offset.

        Applies the transformation: data * factor + offset

        Parameters
        ----------
        factor : float, optional
            Multiplicative factor, default 1.0
        offset : float, optional
            Additive offset, default 0.0

        Returns
        -------
        LazyDfs
            Self for method chaining

        Examples
        --------
        >>> # Convert temperature from Celsius to Fahrenheit
        >>> df.scale(factor=1.8, offset=32.0)

        >>> # Scale by 2 and add 10
        >>> df.scale(factor=2.0, offset=10.0)

        """
        self._operations.append(ScaleOp(factor=factor, offset=offset))
        return self

    def aggregate(
        self, stat: str | Callable[[np.ndarray], np.ndarray] = "mean"
    ) -> LazyDfs:
        """Aggregate over the time dimension.

        Produces a single-timestep output with temporal statistics.

        Parameters
        ----------
        stat : str | Callable, optional
            Either a built-in statistic name ('mean', 'min', 'max', 'median',
            'sum', 'std') or a custom aggregation function that takes a
            2D array (time, space) and returns a 1D array (space).
            Default is 'mean'.

        Returns
        -------
        LazyDfs
            Self for method chaining

        Examples
        --------
        >>> # Temporal mean
        >>> df.aggregate("mean")

        >>> # Temporal maximum
        >>> df.aggregate("max")

        >>> # Custom aggregation
        >>> df.aggregate(lambda x: np.nanpercentile(x, 90, axis=0))

        """
        self._operations.append(AggregateOp(stat=stat))
        return self

    def diff(self, other: str | Path) -> LazyDfs:
        """Compute difference with another file (self - other).

        This operation subtracts another file from the current pipeline output,
        useful for model validation and comparison scenarios.

        Parameters
        ----------
        other : str | Path
            Path to the file to subtract from the current data

        Returns
        -------
        LazyDfs
            Self for method chaining

        Examples
        --------
        >>> # Compare model run with observations
        >>> (scan_dfs("model_results.dfsu")
        ...     .diff("observations.dfsu")
        ...     .to_dfs("model_error.dfsu")
        ... )

        >>> # Compare two model scenarios
        >>> (scan_dfs("scenario_A.dfsu")
        ...     .select([0])
        ...     .diff("scenario_B.dfsu")
        ...     .aggregate(stat="mean")  # Average difference
        ...     .to_dfs("scenario_comparison.dfsu")
        ... )

        """
        self._operations.append(DiffOp(other))
        return self

    def _validate_operations(self, dfs_in: DfsFile) -> None:
        """Validate all operations against file metadata.

        Parameters
        ----------
        dfs_in : DfsFile
            Opened DFS file to validate against

        Raises
        ------
        ValueError
            If any operation is invalid for this file

        """
        file_info = dfs_in.FileInfo
        item_info = dfs_in.ItemInfo
        time_axis = file_info.TimeAxis

        for op in self._operations:
            if isinstance(op, SelectOp):
                # Validate items exist
                try:
                    _valid_item_numbers(item_info, op.items)
                except (IndexError, KeyError) as e:
                    available = [item.Name for item in item_info]
                    raise ValueError(
                        f"Invalid item selection: {e}. Available items: {available}"
                    ) from e

            elif isinstance(op, FilterOp):
                # Validate time range is valid
                try:
                    _TimeInfo.parse(time_axis, op.start, op.end, op.step)
                except ValueError as e:
                    n_steps = time_axis.NumberOfTimeSteps
                    raise ValueError(
                        f"Invalid time filter: {e}. File has {n_steps} timesteps"
                    ) from e

            elif isinstance(op, DiffOp):
                # Validate other file exists and can be opened
                if not op.other_file.exists():
                    raise ValueError(f"Diff file not found: {op.other_file}")
                try:
                    dfs_other = DfsFileFactory.DfsGenericOpen(str(op.other_file))
                    dfs_other.Close()
                except Exception as e:
                    raise ValueError(
                        f"Cannot open diff file {op.other_file}: {e}"
                    ) from e

    def explain(self) -> str:
        """Explain the pipeline execution plan with validation.

        Opens the file, validates all operations, and returns a detailed
        description of what will be executed. This ensures the pipeline
        is valid before execution.

        Returns
        -------
        str
            Detailed explanation of the validated pipeline

        Raises
        ------
        ValueError
            If any operation is invalid for this file

        Examples
        --------
        >>> lazy = (scan_dfs("data.dfsu")
        ...     .select([0, 1])
        ...     .filter(time=slice("2020-01-01", "2020-12-31"))
        ...     .scale(factor=2.0)
        ... )
        >>> print(lazy.explain())

        """
        # Open file to get metadata and validate
        dfs_in = DfsFileFactory.DfsGenericOpen(str(self.path))
        try:
            # Validate all operations
            self._validate_operations(dfs_in)

            # Get file metadata
            file_info = dfs_in.FileInfo
            time_axis = file_info.TimeAxis
            item_names = [item.Name for item in dfs_in.ItemInfo]

            # Build detailed explanation
            lines = [
                "Lazy DFS Pipeline",
                "=" * 50,
                f"Input: {self.path}",
                f"Items ({len(item_names)}): {item_names}",
                f"Timesteps: {time_axis.NumberOfTimeSteps}",
                f"Start time: {time_axis.StartDateTime}",
                "",
                "Operations:",
            ]

            if not self._operations:
                lines.append("  (no operations - will copy all data)")
            else:
                for i, op in enumerate(self._operations, 1):
                    lines.append(f"  {i}. {op.explain()}")

            lines.extend(["", "✓ Pipeline validated successfully"])

            return "\n".join(lines)
        finally:
            dfs_in.Close()

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
            # Validate all operations before processing
            lazy_dfs = LazyDfs(self.infilename)
            lazy_dfs._operations = self.operations
            lazy_dfs._validate_operations(dfs_in)

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
            if op.affects_metadata:
                op.apply(context)

    def _apply_data_operations(self, context: ExecutionContext) -> None:
        """Apply operations that transform data (with_items, rolling)."""
        for op in self.operations:
            if not op.affects_metadata:
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
        # Check for aggregate operation
        if context.aggregate is not None:
            self._process_with_aggregate(context, context.aggregate)
            return

        # Check for rolling operation
        if context.rolling is not None:
            self._process_with_rolling(context, context.rolling)
        else:
            self._process_simple(context)

    def _process_simple(self, context: ExecutionContext) -> None:
        """Process data without rolling (simple passthrough with transforms)."""
        time_info = context.time_info

        # Determine time range
        if time_info is not None:
            start_step = time_info.start_step
            end_step = time_info.end_step
            start_sec = time_info.start_sec
            end_sec = time_info.end_sec
        else:
            start_step = 0
            end_step = context.dfs_in.FileInfo.TimeAxis.NumberOfTimeSteps
            start_sec = -np.inf
            end_sec = np.inf

        # Use the step from context (set by FilterOp)
        step = context.time_step

        # Get transforms if any
        transforms = context.transforms or {}
        scale_params = context.scale_params
        diff_file = context.diff_file

        # Open diff file if needed
        dfs_diff = None
        if diff_file is not None:
            dfs_diff = DfsFileFactory.DfsGenericOpen(str(diff_file))

        try:
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

                    # Apply scale if present
                    if scale_params is not None:
                        factor, offset = scale_params
                        # Convert delete value to NaN before scaling
                        mask = data == context.deletevalue
                        data[mask] = np.nan

                        # Apply scaling
                        data = data * factor + offset

                        # Convert NaN back to delete value
                        data[np.isnan(data)] = context.deletevalue

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

                    # Apply diff if present
                    if dfs_diff is not None:
                        itemdata_diff = dfs_diff.ReadItemTimeStep(
                            item_num + 1, timestep
                        )
                        data_diff = itemdata_diff.Data

                        # Convert delete values to NaN
                        mask_a = data == context.deletevalue
                        mask_b = data_diff == context.deletevalue
                        data[mask_a] = np.nan
                        data_diff[mask_b] = np.nan

                        # Compute difference
                        data = data - data_diff

                        # Convert NaN back to delete value
                        data[np.isnan(data)] = context.deletevalue

                    # Write to output
                    context.dfs_out.WriteItemTimeStep(
                        item_idx + 1, timestep_out, time_sec, data
                    )
        finally:
            if dfs_diff is not None:
                dfs_diff.Close()

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

    def _process_with_aggregate(
        self, context: ExecutionContext, aggregate_op: AggregateOp
    ) -> None:
        """Process data with temporal aggregation using chunked processing.

        For large files, data is processed in spatial chunks to limit memory usage.
        Each chunk contains all timesteps for a subset of spatial elements.
        """
        stat_func = aggregate_op.get_stat_func()
        time_info = context.time_info

        # Determine time range
        if time_info is not None:
            start_step = time_info.start_step
            end_step = time_info.end_step
        else:
            start_step = 0
            end_step = context.dfs_in.FileInfo.TimeAxis.NumberOfTimeSteps

        # Check if this is a layered dfsu (Z coordinate at index 0)
        is_layered = (
            len(context.item_numbers) > 0
            and context.dfs_in.ItemInfo[context.item_numbers[0]].Name == "Z coordinate"
        )

        # Handle Z coordinate for layered dfsu
        if is_layered:
            # Write Z coordinate (first item, first timestep, no aggregation)
            zn_itemdata = context.dfs_in.ReadItemTimeStep(1, start_step)
            context.dfs_out.WriteItemTimeStepNext(0.0, zn_itemdata.Data)
            # Process remaining items (skip Z coordinate)
            item_numbers_to_aggregate = context.item_numbers[1:]
        else:
            item_numbers_to_aggregate = context.item_numbers

        n_items = len(item_numbers_to_aggregate)

        if n_items == 0:
            return

        n_timesteps = end_step - start_step

        # Get data shape from first item to aggregate
        first_itemdata = context.dfs_in.ReadItemTimeStep(
            item_numbers_to_aggregate[0] + 1, start_step
        )
        n_elements = len(first_itemdata.Data)

        # Calculate chunking parameters based on memory constraints
        # Memory per chunk: n_timesteps * chunk_size * n_items * 4 bytes (float32)
        mem_total = 4 * n_timesteps * n_elements * n_items
        n_chunks = max(1, math.ceil(mem_total / self.buffer_size))
        chunk_size = math.ceil(n_elements / n_chunks)

        # Initialize output arrays (full spatial size)
        output_arrays = [
            np.full(n_elements, np.nan, dtype=np.float32) for _ in range(n_items)
        ]

        # Process data in spatial chunks
        e1 = 0
        for _ in range(n_chunks):
            e2 = min(e1 + chunk_size, n_elements)
            actual_chunk_size = e2 - e1

            # Create buffers for this chunk: [n_items][n_timesteps, actual_chunk_size]
            chunk_buffers = [
                np.full((n_timesteps, actual_chunk_size), np.nan, dtype=np.float32)
                for _ in range(n_items)
            ]

            # Read all timesteps for this chunk of elements
            for timestep in range(start_step, end_step):
                timestep_idx = timestep - start_step
                for item_idx, item_num in enumerate(item_numbers_to_aggregate):
                    itemdata = context.dfs_in.ReadItemTimeStep(item_num + 1, timestep)
                    data = itemdata.Data.astype(np.float32)

                    # Replace delete values with NaN
                    data[data == context.deletevalue] = np.nan

                    # Store only the chunk in buffer
                    chunk_buffers[item_idx][timestep_idx, :] = data[e1:e2]

            # Compute aggregate statistic for this chunk
            for item_idx in range(n_items):
                # Apply statistic along time axis (axis=0)
                result = stat_func(chunk_buffers[item_idx], axis=0)

                # Store in output at correct spatial position
                output_arrays[item_idx][e1:e2] = result

            e1 = e2  # Move to next chunk

        # Write results (single timestep at time 0.0)
        for item_idx in range(n_items):
            # Replace NaN with delete value before writing
            output = output_arrays[item_idx]
            output[np.isnan(output)] = context.deletevalue
            context.dfs_out.WriteItemTimeStepNext(0.0, output)


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
