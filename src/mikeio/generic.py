"""Generic functions for working with all types of dfs files."""

from __future__ import annotations

import math
import operator
import os
import pathlib
import warnings
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from shutil import copyfile
from typing import Callable, Literal, Mapping, NamedTuple, Union

import numpy as np
import pandas as pd
from mikecore.DfsBuilder import DfsBuilder
from mikecore.DfsFile import (
    DfsDynamicItemInfo,
    DfsEqCalendarAxis,
    DfsEqTimeAxis,
    DfsFile,
    DfsNonEqCalendarAxis,
    DfsNonEqTimeAxis,
    DfsStaticItem,
    DfsTemporalAxis,
    TimeAxisType,
)
from mikecore.DfsFileFactory import DfsFileFactory
from mikecore.eum import eumQuantity
from tqdm import tqdm, trange

import mikeio

from ._version import __dfs_version__
from ._options import _show_progress
from .dataset import Dataset
from .dfs._dfs import _get_item_info, _opened, _valid_item_numbers
from .eum import EUMType, EUMUnit, ItemInfo
from ._path import normalize_output_path, normalize_path

TimeAxis = Union[
    DfsEqTimeAxis, DfsNonEqTimeAxis, DfsEqCalendarAxis, DfsNonEqCalendarAxis
]

__all__ = [
    "avg_time",
    "concat",
    "diff",
    "extract",
    "fill_corrupt",
    "quantile",
    "scale",
    "add",
    "transform",
    "change_datatype",
]


def _item_names(dfs: DfsFile) -> list[str]:
    """Names of the items in an open dfs file, in order."""
    return [item.Name for item in dfs.ItemInfo]


@dataclass(frozen=True)
class _ChunkInfo:
    n_data: int
    n_chunks: int

    @property
    def chunk_size(self) -> int:
        return math.ceil(self.n_data / self.n_chunks)

    def stop(self, start: int) -> int:
        return min(start + self.chunk_size, self.n_data)

    def chunk_end(self, start: int) -> int:
        e2 = self.stop(start)
        return self.chunk_size - ((start + self.chunk_size) - e2)

    @staticmethod
    def from_dfs(
        dfs: DfsFile, item_numbers: list[int], buffer_size: float
    ) -> _ChunkInfo:
        """Calculate chunk info based on # of elements in dfs file and selected buffer size."""
        n_time_steps = dfs.FileInfo.TimeAxis.NumberOfTimeSteps
        n_data_all: int = np.sum([dfs.ItemInfo[i].ElementCount for i in item_numbers])
        mem_need = 8 * n_time_steps * n_data_all  # n_items *
        n_chunks = math.ceil(mem_need / buffer_size)
        n_data = n_data_all // len(item_numbers)

        return _ChunkInfo(n_data, n_chunks)


@contextmanager
def _clone(
    infilename: str | pathlib.Path,
    outfilename: str | pathlib.Path,
    start_time: datetime | None = None,
    timestep: float | None = None,
    items: Sequence[int | DfsDynamicItemInfo | ItemInfo] | None = None,
    datatype: int | None = None,
    keep_provenance: bool = False,
) -> Iterator[DfsFile]:
    """Create outfilename with the header and static items of infilename.

    Yields the new file, open for writing its dynamic data, and closes it
    when the block exits. The source is closed before the new file is yielded.

    The new file is stamped as written by mikeio, unless `keep_provenance`
    asks for the source's application title and version to be kept.
    """
    infilename = normalize_path(infilename)
    outfilename = normalize_output_path(outfilename)
    with ExitStack() as stack:
        with _opened(DfsFileFactory.DfsGenericOpen(infilename)) as source:
            fi = source.FileInfo

            if keep_provenance:
                builder = DfsBuilder.Create(
                    fi.FileTitle, fi.ApplicationTitle, fi.ApplicationVersion
                )
            else:
                builder = DfsBuilder.Create(fi.FileTitle, "mikeio", __dfs_version__)

            builder.SetDataType(datatype if datatype is not None else fi.DataType)

            builder.SetGeographicalProjection(fi.Projection)

            time_axis = fi.TimeAxis
            if start_time is not None:
                time_axis.StartDateTime = start_time
            if timestep is not None:
                time_axis.TimeStep = timestep
            builder.SetTemporalAxis(time_axis)

            builder.SetItemStatisticsType(fi.StatsType)
            builder.DeleteValueByte = fi.DeleteValueByte
            builder.DeleteValueDouble = fi.DeleteValueDouble
            builder.DeleteValueFloat = fi.DeleteValueFloat
            builder.DeleteValueInt = fi.DeleteValueInt
            builder.DeleteValueUnsignedInt = fi.DeleteValueUnsignedInt

            for customBlock in fi.CustomBlocks:
                builder.AddCustomBlock(customBlock)

            match items:
                case None:
                    for item in source.ItemInfo:
                        builder.AddDynamicItem(item)

                case list():
                    for item in items:
                        match item:
                            case ItemInfo():
                                builder.AddCreateDynamicItem(
                                    name=item.name,
                                    quantity=eumQuantity.Create(item.type, item.unit),
                                    valueType=item.data_value_type,
                                    spatialAxis=source.ItemInfo[-1].SpatialAxis,
                                )
                            case DfsDynamicItemInfo():
                                builder.AddDynamicItem(item)
                            case int():
                                builder.AddDynamicItem(source.ItemInfo[item])

            builder.CreateFile(outfilename)
            # The created file is held by the builder until GetFile() hands it over;
            # hold it from creation, so a failing static item cannot leak it.
            file = stack.enter_context(_opened(builder.DfsFile))

            for static_item in iter(source.ReadStaticItemNext, None):
                builder.AddStaticItem(static_item)

            builder.GetFile()

        yield file


def scale(
    infilename: str | pathlib.Path,
    outfilename: str | pathlib.Path,
    offset: float = 0.0,
    factor: float = 1.0,
    items: Sequence[int | str] | None = None,
) -> None:
    """Apply scaling to any dfs file.

    Parameters
    ----------

    infilename: str | pathlib.Path
        full path to the input file
    outfilename: str | pathlib.Path
        full path to the output file
    offset: float, optional
        value to add to all items, default 0.0
    factor: float, optional
        value to multiply to all items, default 1.0
    items: list[str] or list[int], optional
        Process only selected items, by number (0-based) or name, by default: all

    """
    infilename = normalize_path(infilename)
    outfilename = normalize_output_path(outfilename)
    copyfile(infilename, outfilename)
    with _opened(DfsFileFactory.DfsGenericOpenEdit(outfilename)) as dfs:
        item_numbers = _valid_item_numbers(_item_names(dfs), items)
        n_items = len(item_numbers)

        n_time_steps = dfs.FileInfo.TimeAxis.NumberOfTimeSteps

        deletevalue = dfs.FileInfo.DeleteValueFloat

        for timestep in trange(n_time_steps, disable=not _show_progress()):
            for item in range(n_items):
                itemdata = dfs.ReadItemTimeStep(item_numbers[item] + 1, timestep)
                time = itemdata.Time
                d = itemdata.Data
                d[d == deletevalue] = np.nan

                outdata = d * factor + offset

                outdata[np.isnan(outdata)] = deletevalue
                darray = outdata.astype(np.float32)

                dfs.WriteItemTimeStep(item_numbers[item] + 1, timestep, time, darray)


def fill_corrupt(
    infilename: str | pathlib.Path,
    outfilename: str | pathlib.Path,
    fill_value: float = np.nan,
    items: Sequence[str | int] | None = None,
) -> None:
    """Replace corrupt (unreadable) data with fill_value, default delete value.

    Parameters
    ----------

    infilename: str | pathlib.Path
        full path to the input file
    outfilename: str | pathlib.Path
        full path to the output file
    fill_value: float, optional
        value to use where data is corrupt, default delete value
    items: list[str] or list[int], optional
        Process only selected items, by number (0-based) or name, by default: all

    """
    infilename = normalize_path(infilename)
    outfilename = normalize_output_path(outfilename)
    with _opened(DfsFileFactory.DfsGenericOpen(infilename)) as dfs_i:
        item_numbers = _valid_item_numbers(_item_names(dfs_i), items)
        n_time_steps = dfs_i.FileInfo.TimeAxis.NumberOfTimeSteps
    n_items = len(item_numbers)

    with _clone(infilename, outfilename, items=item_numbers) as dfs:
        deletevalue = dfs.FileInfo.DeleteValueFloat

        reads = [(t, item) for t in range(n_time_steps) for item in range(n_items)]
        next_read = 0
        with tqdm(total=n_time_steps, disable=not _show_progress()) as progress:
            while next_read < len(reads):
                # A failed read leaves the handle unusable: its block is closed
                # and the remaining reads continue from a freshly opened file.
                with _opened(DfsFileFactory.DfsGenericOpen(infilename)) as dfs_i:
                    for timestep, item in reads[next_read:]:
                        next_read += 1
                        itemdata = dfs_i.ReadItemTimeStep(
                            item_numbers[item] + 1, timestep
                        )
                        if itemdata is not None:
                            time = itemdata.Time
                            d = itemdata.Data
                        else:
                            iteminfo: DfsDynamicItemInfo = dfs_i.ItemInfo[item]
                            d = np.zeros(iteminfo.ElementCount)
                            d[:] = fill_value
                            d[d == deletevalue] = np.nan

                        d[np.isnan(d)] = deletevalue
                        darray = d.astype(np.float32)

                        dfs.WriteItemTimeStep(item + 1, timestep, time, darray)

                        if item == n_items - 1:
                            progress.update()
                        if itemdata is None:
                            break


def _process_dfs_files(
    infilename_a: str | pathlib.Path,
    infilename_b: str | pathlib.Path,
    outfilename: str | pathlib.Path,
    op: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> None:
    """Process two dfs files with a specified operation.

    Parameters
    ----------
    infilename_a: str | pathlib.Path
        full path to the first input file
    infilename_b: str | pathlib.Path
        full path to the second input file
    outfilename: str | pathlib.Path
        full path to the output file
    op: Callable[[np.ndarray, np.ndarray], np.ndarray]
        operation to perform on the data arrays

    """
    infilename_a = normalize_path(infilename_a)
    infilename_b = normalize_path(infilename_b)
    outfilename = normalize_output_path(outfilename)
    copyfile(infilename_a, outfilename)

    with (
        _opened(DfsFileFactory.DfsGenericOpen(infilename_a)) as dfs_i_a,
        _opened(DfsFileFactory.DfsGenericOpen(infilename_b)) as dfs_i_b,
        _opened(DfsFileFactory.DfsGenericOpenEdit(outfilename)) as dfs_o,
    ):
        deletevalue = dfs_i_a.FileInfo.DeleteValueFloat

        n_time_steps = dfs_i_a.FileInfo.TimeAxis.NumberOfTimeSteps
        n_items = len(dfs_i_a.ItemInfo)
        # TODO Add checks to verify identical structure of file a and b

        for timestep in trange(n_time_steps, disable=not _show_progress()):
            for item in range(n_items):
                itemdata_a = dfs_i_a.ReadItemTimeStep(item + 1, timestep)
                d_a = itemdata_a.Data
                d_a[d_a == deletevalue] = np.nan

                itemdata_b = dfs_i_b.ReadItemTimeStep(item + 1, timestep)
                d_b = itemdata_b.Data
                d_b[d_b == deletevalue] = np.nan
                time = itemdata_a.Time

                outdata = op(d_a, d_b)

                darray = outdata.astype(np.float32)

                dfs_o.WriteItemTimeStep(item + 1, timestep, time, darray)


# TODO sum is conflicting with the built-in sum function, which we could haved used above.
def sum(
    infilename_a: str | pathlib.Path,
    infilename_b: str | pathlib.Path,
    outfilename: str | pathlib.Path,
) -> None:
    """Add two dfs files (a+b) (deprecated, use add instead)."""
    warnings.warn(FutureWarning("This function is deprecated. Use add instead."))
    _process_dfs_files(infilename_a, infilename_b, outfilename, operator.add)


def add(
    infilename_a: str | pathlib.Path,
    infilename_b: str | pathlib.Path,
    outfilename: str | pathlib.Path,
) -> None:
    """Add two dfs files (a+b)."""
    _process_dfs_files(infilename_a, infilename_b, outfilename, operator.add)


def diff(
    infilename_a: str | pathlib.Path,
    infilename_b: str | pathlib.Path,
    outfilename: str | pathlib.Path,
) -> None:
    """Calculate difference between two dfs files (a-b).

    Parameters
    ----------
    infilename_a: str | pathlib.Path
        full path to the first input file
    infilename_b: str | pathlib.Path
        full path to the second input file
    outfilename: str | pathlib.Path
        full path to the output file

    Examples
    --------
    ```{python}
    from mikeio import generic
    generic.diff("../data/oresundHD_run1.dfsu", "../data/oresundHD_run2.dfsu", "diff.dfsu");
    ```

    """
    _process_dfs_files(infilename_a, infilename_b, outfilename, operator.sub)


def concat(
    infilenames: Sequence[str | pathlib.Path],
    outfilename: str | pathlib.Path,
    keep: Literal["first", "last", "average"] = "last",
) -> None:
    """Concatenates files along the time axis.

    Works on any DFS-format file: dfs0, dfs1, dfs2, dfs3, dfsu, and other
    files in the DFS format such as MIKE 1D result files (res1d). The files
    are joined by timestamp, timestep by timestep, so memory use does not depend on
    file length.

    Overlap handling is defined by the `keep` argument; by default the newer
    file wins in the overlap.

    Parameters
    ----------
    infilenames:
        filenames to concatenate, in chronological order
    outfilename:
        filename of output
    keep:
        how to handle the overlap between consecutive files:

        * 'last' (default): keep the newer file, i.e. take timesteps from a
          file until the next file starts
        * 'first': keep the older file, i.e. take only the timesteps of a file
          that are later than the last timestep already written
        * 'average': average the two files in the overlap; their timestamps
          must be identical there, the later file must not end before the
          earlier one, and no more than two files may overlap at a time

    Raises
    ------
    FileNotFoundError
        If an input file does not exist.
    ValueError
        If the files cannot be concatenated, or if the output file is one of
        the inputs. The message names the file and the property that is the
        problem. Not raised for dfs0 files; see Notes.

    Notes
    -----
    Before any output is written, the inputs are checked:

    * each file starts strictly later than the one before it;
    * there is no gap between consecutive files: on an equidistant time axis
      the next file starts no later than one timestep after the previous
      file ends; on a non-equidistant time axis the files overlap or touch;
    * all files come from the same setup: the same calendar time axis type
      (and timestep duration, if equidistant), the same dynamic items (name,
      quantity, number of elements and data type) and identical static items.

    Files may use different time units. The output has the same time axis
    type as the inputs; non-equidistant inputs keep every selected timestamp
    exactly. Each item keeps its data type. The header and static items are
    taken from the first file, including its application title and version.
    If concatenation fails, no output file is left behind, and an output file
    that existed before is left untouched unless it was already being
    overwritten.

    dfs0 files take a separate path: they are read into a Dataset and
    concatenated with `Dataset.concat`. The checks above do not apply to
    them, and `keep="average"` is not supported for dfs0.

    Examples
    --------
    Merge two overlapping MIKE 1D result files, the second one winning in the
    overlap:

    ```{python}
    from mikeio import generic
    generic.concat(
        ["../data/res1d/network_a.res1d", "../data/res1d/network_b.res1d"], "merged.res1d"
    )
    ```

    """
    infilenames = [normalize_path(f) for f in infilenames]
    # fast path for Dfs0
    suffix = pathlib.Path(infilenames[0]).suffix
    if suffix == ".dfs0":
        dss = [mikeio.read(f) for f in infilenames]
        ds = Dataset.concat(dss, keep=keep)  # type: ignore
        ds.to_dfs(normalize_output_path(outfilename))
        return

    if keep not in ("first", "last", "average"):
        raise ValueError(f"keep must be 'first', 'last' or 'average', not {keep!r}")

    inputs = _read_and_check_inputs(infilenames, keep)
    outfilename = normalize_path(outfilename)
    existed = os.path.exists(outfilename)
    if existed and any(os.path.samefile(f, outfilename) for f in infilenames):
        raise ValueError(f"{outfilename}: the output file is also an input file")

    # On failure, remove the output only if this call created it: a file that
    # existed before is the user's until _clone has overwritten it.
    created = not existed
    # Creates the output file, so only once the inputs are known to be good.
    outfilename = normalize_output_path(outfilename)
    try:
        with _clone(inputs[0].filename, outfilename, keep_provenance=True) as dfs_o:
            created = True
            _write_timesteps(inputs, dfs_o, keep)
    except BaseException:
        if created and os.path.exists(outfilename):
            os.remove(outfilename)
        raise


_Keep = Literal["first", "last", "average"]


@dataclass(frozen=True)
class _ConcatInput:
    """Time axis of one concat input, read before any output is created."""

    filename: str
    axis_start: datetime
    """StartDateTime of the time axis; timestep times are relative to this."""
    start: datetime
    """Time of the first timestep."""
    end: datetime
    """Time of the last timestep."""
    n_timesteps: int
    dt: float | None
    """Timestep duration in seconds, for an equidistant time axis."""
    axis: DfsTemporalAxis
    """The time axis; converts times in the file's time unit to seconds."""

    @staticmethod
    def read(filename: str, dfs: DfsFile) -> _ConcatInput:
        axis = dfs.FileInfo.TimeAxis
        n_timesteps = axis.NumberOfTimeSteps
        # Timestep times are relative to StartDateTime and include the offset.
        start = axis.StartDateTime + timedelta(
            seconds=axis.ToSeconds(axis.StartTimeOffset)
        )
        match axis.TimeAxisType:
            case TimeAxisType.CalendarEquidistant:
                dt = axis.TimeStepInSeconds()
                end = start + timedelta(seconds=dt * (n_timesteps - 1))
            case TimeAxisType.CalendarNonEquidistant:
                dt = None
                end = start + timedelta(seconds=axis.ToSeconds(axis.TimeSpan))
            case other:
                raise ValueError(
                    f"{filename}: time axis type {other.name} is not supported; "
                    "concat needs a calendar time axis"
                )
        return _ConcatInput(
            filename, axis.StartDateTime, start, end, n_timesteps, dt, axis
        )


class _ItemSignature(NamedTuple):
    name: str
    eum_type: int
    eum_unit: int
    element_count: int
    data_type: int

    @staticmethod
    def of(item: DfsDynamicItemInfo) -> _ItemSignature:
        return _ItemSignature(
            item.Name,
            int(item.Quantity.Item),
            int(item.Quantity.Unit),
            item.ElementCount,
            int(item.DataType),
        )


class _StaticItemSignature(NamedTuple):
    name: str
    data_type: int
    element_count: int
    data: bytes

    @staticmethod
    def of(item: DfsStaticItem) -> _StaticItemSignature:
        return _StaticItemSignature(
            item.Name, int(item.DataType), item.ElementCount, item.Data.tobytes()
        )


@dataclass(frozen=True)
class _ModelSetup:
    """What all concat inputs must share: time axis type, items, static items."""

    filename: str
    axis_type: TimeAxisType
    items: list[_ItemSignature]
    static_items: list[_StaticItemSignature]

    @staticmethod
    def read(filename: str, dfs: DfsFile) -> _ModelSetup:
        return _ModelSetup(
            filename,
            dfs.FileInfo.TimeAxis.TimeAxisType,
            [_ItemSignature.of(it) for it in dfs.ItemInfo],
            [_StaticItemSignature.of(s) for s in iter(dfs.ReadStaticItemNext, None)],
        )

    def check_same(self, filename: str, dfs: DfsFile) -> None:
        """Raise if `dfs` does not have this setup."""
        axis_type = dfs.FileInfo.TimeAxis.TimeAxisType
        if axis_type != self.axis_type:
            raise ValueError(
                f"{filename}: time axis type {axis_type.name} differs from "
                f"{self.axis_type.name} in {self.filename}"
            )

        items = [_ItemSignature.of(it) for it in dfs.ItemInfo]
        if len(items) != len(self.items):
            raise ValueError(
                f"{filename}: has {len(items)} items, "
                f"{self.filename} has {len(self.items)}"
            )
        for i, (expected, item) in enumerate(zip(self.items, items)):
            for field in _ItemSignature._fields:
                value, expected_value = getattr(item, field), getattr(expected, field)
                if value != expected_value:
                    raise ValueError(
                        f"{filename}: item {i + 1} ({item.name}) has "
                        f"{field.replace('_', ' ')} {value}, "
                        f"{self.filename} has {expected_value}"
                    )

        n_static = 0
        for static in iter(dfs.ReadStaticItemNext, None):
            if n_static >= len(self.static_items):
                raise ValueError(
                    f"{filename}: has more static items than {self.filename} "
                    f"({len(self.static_items)})"
                )
            if _StaticItemSignature.of(static) != self.static_items[n_static]:
                raise ValueError(
                    f"{filename}: static item {n_static + 1} ({static.Name}) "
                    f"differs from {self.filename}"
                )
            n_static += 1
        if n_static != len(self.static_items):
            raise ValueError(
                f"{filename}: has {n_static} static items, "
                f"{self.filename} has {len(self.static_items)}"
            )


def _read_and_check_inputs(filenames: Sequence[str], keep: _Keep) -> list[_ConcatInput]:
    """Read the inputs' headers and check order, continuity and same setup."""
    for f in filenames:
        if not os.path.isfile(f):
            raise FileNotFoundError(f"Input file not found: {f}")

    with _opened(DfsFileFactory.DfsGenericOpen(filenames[0])) as dfs:
        setup = _ModelSetup.read(filenames[0], dfs)
        inputs = [_ConcatInput.read(filenames[0], dfs)]

    for f in filenames[1:]:
        with _opened(DfsFileFactory.DfsGenericOpen(f)) as dfs:
            setup.check_same(f, dfs)
            inputs.append(_ConcatInput.read(f, dfs))

    for i, (prev, cur) in enumerate(zip(inputs, inputs[1:])):
        if prev.dt != cur.dt:
            raise ValueError(
                f"{cur.filename}: timestep {cur.dt} s differs from "
                f"{prev.dt} s in {prev.filename}"
            )
        if cur.start <= prev.start:
            raise ValueError(
                f"{cur.filename}: starts at {cur.start}, which is not after "
                f"the start of the previous file {prev.filename} ({prev.start}). "
                "Files must be in chronological order."
            )
        latest_start = prev.end + timedelta(seconds=prev.dt or 0.0)
        if cur.start > latest_start:
            raise ValueError(
                f"{cur.filename}: starts at {cur.start}, after the end of the "
                f"previous file {prev.filename} ({prev.end}). "
                "Gap in time axis detected - not supported"
            )
        if keep == "average" and cur.end < prev.end:
            raise ValueError(
                f"{cur.filename}: lies entirely inside {prev.filename} "
                f"({cur.start} to {cur.end} is within {prev.start} to "
                f"{prev.end}). keep='average' needs the later file to "
                "end no earlier than the previous one."
            )
        if keep == "average" and i > 0 and cur.start <= inputs[i - 1].end:
            raise ValueError(
                f"{cur.filename}: starts at {cur.start}, before the end of "
                f"{inputs[i - 1].filename} ({inputs[i - 1].end}). "
                "keep='average' supports overlaps of two files only."
            )
    return inputs


class _TimestepReader:
    """Reads the timesteps of one open concat input, and their absolute times.

    On an equidistant axis a timestep's time is computed. On a non-equidistant
    axis it is read with the first item, whose data is then kept for the
    timestep so that it is not read twice.
    """

    def __init__(self, dfs: DfsFile, info: _ConcatInput) -> None:
        self.info = info
        self._dfs = dfs
        self._n_items = len(dfs.ItemInfo)
        self._timestep = -1
        self._first_item: np.ndarray | None = None

    def time(self, timestep: int) -> datetime:
        info = self.info
        if info.dt is not None:
            return info.start + timedelta(seconds=timestep * info.dt)
        itemdata = self._dfs.ReadItemTimeStep(1, timestep)
        self._timestep, self._first_item = timestep, itemdata.Data
        return info.axis_start + timedelta(seconds=info.axis.ToSeconds(itemdata.Time))

    def first_timestep_after(self, time: datetime) -> int:
        timestep = 0
        while timestep < self.info.n_timesteps and self.time(timestep) <= time:
            timestep += 1
        return timestep

    def read(self, timestep: int) -> list[np.ndarray]:
        dfs = self._dfs
        if timestep == self._timestep and self._first_item is not None:
            first_item = self._first_item
        else:
            first_item = dfs.ReadItemTimeStep(1, timestep).Data
        return [first_item] + [
            dfs.ReadItemTimeStep(item + 1, timestep).Data
            for item in range(1, self._n_items)
        ]


def _write_timesteps(inputs: list[_ConcatInput], dfs_o: DfsFile, keep: _Keep) -> None:
    """Write the timesteps that `keep` selects from the inputs to dfs_o."""
    out_start = inputs[0].axis_start
    out_axis = inputs[0].axis

    with ExitStack() as prev_file:

        def write(time: datetime, data: list[np.ndarray]) -> None:
            time_out = out_axis.ToRelativeTime((time - out_start).total_seconds())
            for d in data:
                dfs_o.WriteItemTimeStepNext(time_out, d)

        last_written: datetime | None = None
        overlap: tuple[_TimestepReader, int] | None = None  # for keep="average"

        for i, info in enumerate(tqdm(inputs, disable=not _show_progress())):
            next_start = inputs[i + 1].start if i + 1 < len(inputs) else None
            with ExitStack() as this_file:
                reader = _TimestepReader(
                    this_file.enter_context(
                        _opened(DfsFileFactory.DfsGenericOpen(info.filename))
                    ),
                    info,
                )
                timestep = 0
                if keep == "first" and last_written is not None:
                    timestep = reader.first_timestep_after(last_written)
                elif overlap is not None:
                    timestep = _write_average(*overlap, reader, write)
                    prev_file.close()

                while timestep < info.n_timesteps:
                    time = reader.time(timestep)
                    if keep != "first" and next_start is not None:
                        if time >= next_start:
                            break
                    write(time, reader.read(timestep))
                    last_written = time
                    timestep += 1

                if keep == "average" and next_start is not None:
                    overlap = (reader, timestep)
                    prev_file.push(this_file.pop_all())


def _write_average(
    prev: _TimestepReader,
    prev_timestep: int,
    cur: _TimestepReader,
    write: Callable[[datetime, list[np.ndarray]], None],
) -> int:
    """Write the average of the overlap of two inputs; return cur's next timestep."""
    cur_timestep = 0
    for prev_timestep in range(prev_timestep, prev.info.n_timesteps):
        time = prev.time(prev_timestep)
        if cur_timestep >= cur.info.n_timesteps or cur.time(cur_timestep) != time:
            raise ValueError(
                f"{cur.info.filename}: has no timestep at {time}, which is in the "
                f"overlap with {prev.info.filename}. keep='average' needs identical "
                "timestamps in the overlap."
            )
        data = [
            (prev_data * 0.5 + cur_data * 0.5).astype(prev_data.dtype, copy=False)
            for prev_data, cur_data in zip(
                prev.read(prev_timestep), cur.read(cur_timestep)
            )
        ]
        write(time, data)
        cur_timestep += 1
    return cur_timestep


def extract(
    infilename: str | pathlib.Path,
    outfilename: str | pathlib.Path,
    start: int | float | str | datetime = 0,
    end: int | float | str | datetime = -1,
    step: int = 1,
    items: Sequence[int | str] | None = None,
) -> None:
    """Extract timesteps and/or items to a new dfs file.

    Parameters
    ----------
    infilename : str | pathlib.Path
        path to input dfs file
    outfilename : str | pathlib.Path
        path to output dfs file
    start : int, float, str or datetime, optional
        start of extraction as either step, relative seconds
        or datetime/str, by default 0 (start of file)
    end : int, float, str or datetime, optional
        end of extraction as either step, relative seconds
        or datetime/str, by default -1 (end of file)
    step : int, optional
        jump this many step, by default 1 (every step between start and end)
    items : int, list(int), str, list(str), optional
        items to be extracted to new file

    Examples
    --------
    >>> extract('f_in.dfs0', 'f_out.dfs0', start='2018-1-1')
    >>> extract('f_in.dfs2', 'f_out.dfs2', end=-3)
    >>> extract('f_in.dfsu', 'f_out.dfsu', start=1800.0, end=3600.0)
    >>> extract('f_hourly.dfsu', 'f_daily.dfsu', step=24)
    >>> extract('f_in.dfsu', 'f_out.dfsu', items=[2, 0])
    >>> extract('f_in.dfsu', 'f_out.dfsu', items="Salinity")
    >>> extract('f_in.dfsu', 'f_out.dfsu', end='2018-2-1 00:00', items="Salinity")

    """
    infilename = normalize_path(infilename)
    outfilename = normalize_output_path(outfilename)
    with _opened(DfsFileFactory.DfsGenericOpenEdit(infilename)) as dfs_i:
        is_layered_dfsu = dfs_i.ItemInfo[0].Name == "Z coordinate"

        time = _TimeInfo.parse(dfs_i.FileInfo.TimeAxis, start, end, step)
        item_numbers = _valid_item_numbers(
            _item_names(dfs_i), items, ignore_first=is_layered_dfsu
        )

        if is_layered_dfsu:
            item_numbers = [it + 1 for it in item_numbers]
            item_numbers.insert(0, 0)

        with _clone(
            infilename,
            outfilename,
            start_time=time.file_start_new,
            timestep=time.timestep,
            items=item_numbers,
        ) as dfs_o:
            file_start_shift = 0
            if time.file_start_new is not None:
                file_start_orig = dfs_i.FileInfo.TimeAxis.StartDateTime
                file_start_shift = (
                    time.file_start_new - file_start_orig
                ).total_seconds()

            timestep_out = -1
            for timestep in range(time.start_step, time.end_step, step):
                for item_out, item in enumerate(item_numbers):
                    itemdata = dfs_i.ReadItemTimeStep((item + 1), timestep)
                    time_sec = itemdata.Time

                    if time_sec > time.end_sec:
                        return

                    if time_sec >= time.start_sec:
                        if item == item_numbers[0]:
                            timestep_out = timestep_out + 1
                        time_sec_out = time_sec - file_start_shift

                        outdata = itemdata.Data
                        dfs_o.WriteItemTimeStep(
                            (item_out + 1), timestep_out, time_sec_out, outdata
                        )


@dataclass
class _TimeInfo:
    """Parsed time information.

    Attributes
    ----------
    file_start_new : datetime | None
        new start time for the new file
    start_step : int
        start step
    start_sec : float
        start time in seconds
    end_step : int
        end step
    end_sec : float
        end time in seconds
    timestep : float | None
        timestep in seconds

    """

    file_start_new: datetime | None
    start_step: int
    start_sec: float
    end_step: int
    end_sec: float
    timestep: float | None

    @staticmethod
    def parse(
        time_axis: TimeAxis,
        start: int | float | str | datetime,
        end: int | float | str | datetime,
        step: int,
    ) -> _TimeInfo:
        """Helper function for parsing start and end arguments."""
        n_time_steps = time_axis.NumberOfTimeSteps
        file_start_datetime = time_axis.StartDateTime
        file_start_sec = time_axis.StartTimeOffset
        start_sec = file_start_sec

        timespan = 0
        if time_axis.TimeAxisType == 3:
            timespan = time_axis.TimeStep * (n_time_steps - 1)
        elif time_axis.TimeAxisType == 4:
            timespan = time_axis.TimeSpan
        else:
            raise ValueError("TimeAxisType not supported")

        file_end_sec = start_sec + timespan
        end_sec = file_end_sec

        start_step = 0
        if isinstance(start, int):
            start_step = start
        elif isinstance(start, float):
            start_sec = start
        elif isinstance(start, str):
            parts = start.split(",")
            start = parts[0]
            if len(parts) == 2:
                end = parts[1]
            start = pd.to_datetime(start)

        if isinstance(start, datetime):
            start_sec = (start - file_start_datetime).total_seconds()

        end_step = n_time_steps
        if isinstance(end, int):
            if end < 0:
                end = end_step + end + 1
            end_step = end
        elif isinstance(end, float):
            end_sec = end
        elif isinstance(end, str):
            end = pd.to_datetime(end)

        if isinstance(end, datetime):
            end_sec = (end - file_start_datetime).total_seconds()

        if start_step < 0:
            raise ValueError(
                f"start cannot be before start of file. start={start_step} is invalid"
            )

        if start_sec < file_start_sec:
            raise ValueError(
                f"start cannot be before start of file start={start_step} is invalid"
            )

        if (end_sec < start_sec) or (end_step < start_step):
            raise ValueError("end must be after start")

        if end_step > n_time_steps:
            raise ValueError(
                f"end cannot be after end of file. end={end_step} is invalid."
            )

        if end_sec > file_end_sec:
            raise ValueError(
                f"end cannot be after end of file. end={end_sec} is invalid."
            )

        file_start_new = None
        if time_axis.TimeAxisType == TimeAxisType.CalendarEquidistant:
            dt = time_axis.TimeStep
            if (start_sec > file_start_sec) and (start_step == 0):
                # we can find the coresponding step
                start_step = int((start_sec - file_start_sec) / dt)
            file_start_new = file_start_datetime + timedelta(seconds=start_step * dt)
        elif time_axis.TimeAxisType == TimeAxisType.CalendarNonEquidistant:
            if start_sec > file_start_sec:
                file_start_new = file_start_datetime + timedelta(seconds=start_sec)

        timestep = _TimeInfo._parse_step(time_axis, step)

        return _TimeInfo(
            file_start_new, start_step, start_sec, end_step, end_sec, timestep
        )

    @staticmethod
    def _parse_step(time_axis: TimeAxis, step: int) -> float | None:
        """Helper function for parsing step argument."""
        if step == 1:
            timestep = None
        elif time_axis.TimeAxisType == TimeAxisType.CalendarEquidistant:
            timestep = time_axis.TimeStep * step
        elif time_axis.TimeAxisType == TimeAxisType.CalendarNonEquidistant:
            timestep = None
        else:
            raise ValueError("TimeAxisType not supported")
        return timestep


def avg_time(
    infilename: str | pathlib.Path,
    outfilename: str | pathlib.Path,
    skipna: bool = True,
) -> None:
    """Create a temporally averaged dfs file.

    Parameters
    ----------
    infilename : str | pathlib.Path
        input filename
    outfilename : str | pathlib.Path
        output filename
    skipna : bool, optional
        exclude NaN/delete values when computing the result, default True

    """
    infilename = normalize_path(infilename)
    outfilename = normalize_output_path(outfilename)
    with (
        _opened(DfsFileFactory.DfsGenericOpen(infilename)) as dfs_i,
        _clone(infilename, outfilename) as dfs_o,
    ):
        n_items = len(dfs_i.ItemInfo)
        item_numbers = list(range(n_items))

        n_time_steps = dfs_i.FileInfo.TimeAxis.NumberOfTimeSteps
        deletevalue = dfs_i.FileInfo.DeleteValueFloat

        outdatalist = []
        steps_list = []

        # step 0
        for item in item_numbers:
            indatatime = dfs_i.ReadItemTimeStep(item + 1, 0.0)
            indata = indatatime.Data
            has_value = indata != deletevalue
            indata[~has_value] = np.nan
            outdatalist.append(indata.astype(np.float32))
            step0 = np.zeros_like(indata, dtype=np.int32)
            step0[has_value] = 1
            steps_list.append(step0)

        for timestep in trange(1, n_time_steps, disable=not _show_progress()):
            for item in range(n_items):
                itemdata = dfs_i.ReadItemTimeStep(item_numbers[item] + 1, timestep)
                d = itemdata.Data
                has_value = d != deletevalue

                outdatalist[item][has_value] += d[has_value]
                steps_list[item][has_value] += 1

        for item in range(n_items):
            darray = np.zeros_like(outdatalist[item], dtype=np.float32)
            if skipna:
                has_value = steps_list[item] == n_time_steps
            else:
                has_value = steps_list[item] > 0
            darray[has_value] = outdatalist[item][has_value].astype(
                np.float32
            ) / steps_list[item][has_value].astype(np.float32)
            darray[~has_value] = deletevalue
            dfs_o.WriteItemTimeStepNext(0.0, darray)


def quantile(
    infilename: str | pathlib.Path,
    outfilename: str | pathlib.Path,
    q: float | Sequence[float],
    *,
    items: Sequence[int | str] | int | str | None = None,
    skipna: bool = True,
    buffer_size: float = 1.0e9,
) -> None:
    """Create temporal quantiles of all items in dfs file.

    Parameters
    ----------
    infilename : str | pathlib.Path
        input filename
    outfilename : str | pathlib.Path
        output filename
    q: array_like of float
        Quantile or sequence of quantiles to compute,
        which must be between 0 and 1 inclusive.
    items: list[str] or list[int], optional
        Process only selected items, by number (0-based) or name, by default: all
    skipna : bool, optional
        exclude NaN/delete values when computing the result, default True
    buffer_size: float, optional
        for huge files the quantiles need to be calculated for chunks of
        elements. buffer_size gives the maximum amount of memory available
        for the computation in bytes, by default 1e9 (=1GB)

    Examples
    --------
    >>> quantile("in.dfsu", "IQR.dfsu", q=[0.25,0.75])

    >>> quantile("huge.dfsu", "Q01.dfsu", q=0.1, buffer_size=5.0e9)

    >>> quantile("with_nans.dfsu", "Q05.dfsu", q=0.5, skipna=False)

    """
    infilename = normalize_path(infilename)
    outfilename = normalize_output_path(outfilename)
    func = np.nanquantile if skipna else np.quantile

    with _opened(DfsFileFactory.DfsGenericOpen(infilename)) as dfs_i:
        is_dfsu_3d = dfs_i.ItemInfo[0].Name == "Z coordinate"

        item_numbers = _valid_item_numbers(_item_names(dfs_i), items)

        if is_dfsu_3d and 0 in item_numbers:
            item_numbers.remove(0)  # Remove Zn item for special treatment

        n_time_steps = dfs_i.FileInfo.TimeAxis.NumberOfTimeSteps

        # TODO: better handling of different item sizes (zn...)

        ci = _ChunkInfo.from_dfs(dfs_i, item_numbers, buffer_size)

        qvec: Sequence[float] = [q] if isinstance(q, float) else q
        qtxt = [f"Quantile {q!r}" for q in qvec]
        core_items = [dfs_i.ItemInfo[i] for i in item_numbers]
        out_items = _get_repeated_items(core_items, prefixes=qtxt)

        if is_dfsu_3d:
            out_items.insert(0, dfs_i.ItemInfo[0])

        with _clone(infilename, outfilename, items=out_items) as dfs_o:
            n_items_out = len(out_items)
            if is_dfsu_3d:
                n_items_out = n_items_out - 1

            datalist = []
            outdatalist = []

            for item in item_numbers:
                indata = _read_item(dfs_i, item, 0)
                for _ in qvec:
                    outdatalist.append(np.zeros_like(indata))
                datalist.append(np.zeros((n_time_steps, ci.chunk_size)))

            e1 = 0
            for _ in range(ci.n_chunks):
                e2 = ci.stop(e1)
                # the last chunk may be smaller than the rest
                chunk_end = ci.chunk_end(e1)

                _read_items_into_chunk(
                    dfs_i, item_numbers, n_time_steps, e1, e2, chunk_end, datalist
                )
                _compute_quantiles_for_chunk(
                    datalist, outdatalist, qvec, func, ci.chunk_size, e1, e2, chunk_end
                )

                e1 = e2

            if is_dfsu_3d:
                znitemdata = dfs_i.ReadItemTimeStep(1, 0)
                # TODO should this be static Z coordinates instead?
                dfs_o.WriteItemTimeStepNext(0.0, znitemdata.Data)

            for item in range(n_items_out):
                darray = outdatalist[item].astype(np.float32)
                dfs_o.WriteItemTimeStepNext(0.0, darray)


def _read_items_into_chunk(
    dfs_i: DfsFile,
    item_numbers: Sequence[int],
    n_time_steps: int,
    e1: int,
    e2: int,
    chunk_end: int,
    datalist: list[np.ndarray],
) -> None:
    for timestep in range(n_time_steps):
        for item_out, item_no in enumerate(item_numbers):
            itemdata = _read_item(dfs_i, item_no, timestep)
            datalist[item_out][timestep, 0:chunk_end] = itemdata[e1:e2]


def _compute_quantiles_for_chunk(
    datalist: Sequence[np.ndarray],
    outdatalist: Sequence[np.ndarray],
    qvec: Sequence[float],
    func: Callable[..., np.ndarray],
    chunk_size: int,
    e1: int,
    e2: int,
    chunk_end: int,
) -> None:
    for item in range(len(datalist)):
        qdat = np.zeros((len(qvec), chunk_size))
        qdat[:, :] = func(datalist[item][:, 0:chunk_end], q=qvec, axis=0)
        for j in range(len(qvec)):
            outdatalist[item * len(qvec) + j][e1:e2] = qdat[j, :]


def _read_item(dfs: DfsFile, item: int, timestep: int) -> np.ndarray:
    indatatime = dfs.ReadItemTimeStep(item + 1, timestepIndex=timestep)
    indata = indatatime.Data
    has_value = indata != dfs.FileInfo.DeleteValueFloat
    indata[~has_value] = np.nan

    return indata.astype(np.float64)


def _get_repeated_items(
    items_in: list[DfsDynamicItemInfo], prefixes: list[str]
) -> list[ItemInfo]:
    item_numbers = _valid_item_numbers([item.Name for item in items_in])
    items_in = _get_item_info(items_in)

    new_items = []
    for item_num in item_numbers:
        for prefix in prefixes:
            item = deepcopy(items_in[item_num])
            item.name = f"{prefix}, {item.name}"
            new_items.append(item)

    return new_items


def change_datatype(
    infilename: str | pathlib.Path,
    outfilename: str | pathlib.Path,
    datatype: int,
) -> None:
    """Change datatype of a DFS file.

    The data type tag is used to classify the file within a specific modeling context,
    such as MIKE 21. There is no global standard for these tags—they are interpreted
    locally within a model setup.

    Application developers can use these tags to classify files such as
    bathymetries, input data, or result files according to their own conventions.

    Default data type values assigned by MikeIO when creating new files are:
    - dfs0: datatype=1
    - dfs1-3: datatype=0
    - dfsu: datatype=2001

    Parameters
    ----------
    infilename : str | pathlib.Path
        input filename
    outfilename : str | pathlib.Path
        output filename
    datatype: int
        DataType to be used for the output file

    Examples
    --------
    >>> change_datatype("in.dfsu", "out.dfsu", datatype=107)

    """
    infilename = normalize_path(infilename)
    outfilename = normalize_output_path(outfilename)
    with (
        _clone(infilename, outfilename, datatype=datatype) as dfs_out,
        _opened(DfsFileFactory.DfsGenericOpen(infilename)) as dfs_in,
    ):
        sourceData = dfs_in.ReadItemTimeStepNext()
        while sourceData:
            dfs_out.WriteItemTimeStepNext(sourceData.Time, sourceData.Data)
            sourceData = dfs_in.ReadItemTimeStepNext()


class DerivedItem:
    """Item derived from other items.

    Parameters
    ----------
    item: ItemInfo
        ItemInfo object for the derived item
    func: Callable[[Mapping[str, np.ndarray]], np.ndarray] | None
        Function to compute the derived item from a mapping of item names to data arrays.
        If None, the item data will be returned directly from the mapping using the item's name.
        Default is None.

    Example
    -------
    ```{python}
    import numpy as np
    import mikeio
    from mikeio.generic import DerivedItem

    item = DerivedItem(
            item=ItemInfo("Current Speed", mikeio.EUMType.Current_Speed),
            func=lambda x: np.sqrt(x["U velocity"] ** 2 + x["V velocity"] ** 2),
        )

    """

    def __init__(
        self,
        name: str,
        type: EUMType | None = None,
        unit: EUMUnit | None = None,
        func: Callable[[Mapping[str, np.ndarray]], np.ndarray] | None = None,
    ) -> None:
        """Create a DerivedItem.

        Parameters
        ----------
        name: str
            Name of the derived item.
        type: EUMType
            EUMType of the derived item.
        unit: EUMUnit | None, optional
            EUMUnit of the derived item, pass None to use the default unit for the type.
            Default is None.
        func: Callable[[Mapping[str, np.ndarray]], np.ndarray] | None, optional
            Function to compute the derived item from a mapping of item names to data arrays.

        """
        self.item = ItemInfo(name, type, unit)
        self.func = func


def transform(
    infilename: str | pathlib.Path,
    outfilename: str | pathlib.Path,
    vars: Sequence[DerivedItem],
    keep_existing_items: bool = True,
) -> None:
    """Transform a dfs file by applying functions to items.

    Parameters
    ----------
    infilename: str | pathlib.Path
        full path to the input file
    outfilename: str | pathlib.Path
        full path to the output file
    vars: Sequence[DerivedItem]
        List of derived items to compute.
    keep_existing_items: bool, optional
        If True, existing items in the input file will be kept in the output file.
        If False, only the derived items will be written to the output file.
        Default is True.

    Examples
    --------
    ```{python}
    import numpy as np
    import mikeio
    from mikeio.generic import DerivedItem, transform
    item = DerivedItem(
            name="Current Speed",
            type=mikeio.EUMType.Current_Speed,
            func=lambda x: np.sqrt(x["U velocity"] ** 2 + x["V velocity"] ** 2),
        )
    transform(
        infilename="../data/oresundHD_run1.dfsu",
        outfilename="out.dfsu",
        vars=[item],
        keep_existing_items=False,
    )
    ```

    """
    infilename = normalize_path(infilename)
    outfilename = normalize_output_path(outfilename)
    remove_output = False
    try:
        with _opened(DfsFileFactory.DfsGenericOpen(infilename)) as dfs_i:
            item_numbers = _valid_item_numbers(_item_names(dfs_i))
            n_items = len(item_numbers)

            items = [v.item for v in vars]
            funcs = {v.item.name: v.func for v in vars}

            if keep_existing_items:
                existing_items = [
                    ItemInfo.from_mikecore_dynamic_item_info(
                        dfs_i.ItemInfo[i],
                    )
                    for i in item_numbers
                ]
                items = existing_items + items

            with _clone(infilename, outfilename, items=items) as dfs:
                n_time_steps = dfs_i.FileInfo.TimeAxis.NumberOfTimeSteps

                for timestep in range(n_time_steps):
                    data = {}
                    for item in range(n_items):
                        name = dfs_i.ItemInfo[item].Name
                        data[name] = dfs_i.ReadItemTimeStep(
                            item_numbers[item] + 1, timestep
                        ).Data

                    for out_item in items:
                        func = funcs.get(out_item.name, None)
                        if func is None:
                            darray = data[out_item.name]
                        else:
                            try:
                                darray = func(data)
                            except KeyError as e:
                                missing_key = e.args[0]
                                keys = ", ".join(data.keys())
                                remove_output = True
                                raise KeyError(
                                    f"Item '{missing_key}' is not available in the file. Available items: {keys}"
                                )
                        dfs.WriteItemTimeStepNext(0.0, darray)
    finally:
        if remove_output:
            pathlib.Path(outfilename).unlink()
