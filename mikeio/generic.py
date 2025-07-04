"""Generic functions for working with all types of dfs files."""

from __future__ import annotations

import math
import operator
import os
import pathlib
import warnings
from collections.abc import Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from shutil import copyfile
from typing import Callable, Mapping, Union

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
    TimeAxisType,
)
from mikecore.DfsFileFactory import DfsFileFactory
from mikecore.eum import eumQuantity
from tqdm import tqdm, trange

import mikeio

from . import __dfs_version__
from .dfs._dfs import _get_item_info, _valid_item_numbers
from .eum import EUMType, EUMUnit, ItemInfo

TimeAxis = Union[
    DfsEqTimeAxis, DfsNonEqTimeAxis, DfsEqCalendarAxis, DfsNonEqCalendarAxis
]

show_progress = True

__all__ = [
    "avg_time",
    "concat",
    "diff",
    "extract",
    "fill_corrupt",
    "quantile",
    "scale",
    "sum",
    "change_datatype",
]


@dataclass
class _ChunkInfo:
    def __init__(self, n_data: int, n_chunks: int):
        self.n_data = n_data
        self.n_chunks = n_chunks

    def __repr__(self) -> str:
        return f"_ChunkInfo(n_chunks={self.n_chunks}, n_data={self.n_data}, chunk_size={self.chunk_size})"

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


def _clone(
    infilename: str | pathlib.Path,
    outfilename: str | pathlib.Path,
    start_time: datetime | None = None,
    timestep: float | None = None,
    items: Sequence[int | DfsDynamicItemInfo | ItemInfo] | None = None,
    datatype: int | None = None,
) -> DfsFile:
    source = DfsFileFactory.DfsGenericOpen(str(infilename))
    fi = source.FileInfo

    builder = DfsBuilder.Create(fi.FileTitle, "mikeio", __dfs_version__)

    # Set up the header
    if datatype is None:
        builder.SetDataType(fi.DataType)
    else:
        builder.SetDataType(datatype)

    builder.SetGeographicalProjection(fi.Projection)

    # Copy time axis
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

    # Copy custom blocks - if any
    for customBlock in fi.CustomBlocks:
        builder.AddCustomBlock(customBlock)

    if isinstance(items, Iterable) and not isinstance(items, str):
        for item in items:
            if isinstance(item, ItemInfo):
                builder.AddCreateDynamicItem(
                    item.name,
                    eumQuantity.Create(item.type, item.unit),
                    spatialAxis=source.ItemInfo[-1].SpatialAxis,
                )
            elif isinstance(item, DfsDynamicItemInfo):
                builder.AddDynamicItem(item)
            elif isinstance(item, int):
                builder.AddDynamicItem(source.ItemInfo[item])

    elif isinstance(items, (int)) or items is None:
        # must be str/int refering to original file (or None)
        item_numbers = _valid_item_numbers(source.ItemInfo, items)
        items = [source.ItemInfo[item] for item in item_numbers]
        for item in items:
            builder.AddDynamicItem(item)
    else:
        raise ValueError("Items of type: {type(items)} is not supported")

    builder.CreateFile(str(outfilename))

    for static_item in iter(source.ReadStaticItemNext, None):
        builder.AddStaticItem(static_item)

    file = builder.GetFile()

    source.Close()

    return file


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
    infilename = str(infilename)
    outfilename = str(outfilename)
    copyfile(infilename, outfilename)
    dfs = DfsFileFactory.DfsGenericOpenEdit(outfilename)

    item_numbers = _valid_item_numbers(dfs.ItemInfo, items)
    n_items = len(item_numbers)

    n_time_steps = dfs.FileInfo.TimeAxis.NumberOfTimeSteps

    deletevalue = dfs.FileInfo.DeleteValueFloat

    for timestep in trange(n_time_steps, disable=not show_progress):
        for item in range(n_items):
            itemdata = dfs.ReadItemTimeStep(item_numbers[item] + 1, timestep)
            time = itemdata.Time
            d = itemdata.Data
            d[d == deletevalue] = np.nan

            outdata = d * factor + offset

            outdata[np.isnan(outdata)] = deletevalue
            darray = outdata.astype(np.float32)

            dfs.WriteItemTimeStep(item_numbers[item] + 1, timestep, time, darray)

    dfs.Close()


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
    dfs_i = DfsFileFactory.DfsGenericOpen(infilename)

    item_numbers = _valid_item_numbers(dfs_i.ItemInfo, items)
    n_items = len(item_numbers)

    dfs = _clone(
        str(infilename),
        str(outfilename),
        items=item_numbers,
    )

    n_time_steps = dfs_i.FileInfo.TimeAxis.NumberOfTimeSteps

    deletevalue = dfs.FileInfo.DeleteValueFloat

    for timestep in trange(n_time_steps, disable=not show_progress):
        for item in range(n_items):
            itemdata = dfs_i.ReadItemTimeStep(item_numbers[item] + 1, timestep)
            if itemdata is not None:
                time = itemdata.Time
                d = itemdata.Data
            else:
                iteminfo: DfsDynamicItemInfo = dfs_i.ItemInfo[item]
                d = np.zeros(iteminfo.ElementCount)
                d[:] = fill_value
                d[d == deletevalue] = np.nan

                # close and re-open file to solve problem with reading
                dfs_i.Close()
                dfs_i = DfsFileFactory.DfsGenericOpen(infilename)

            d[np.isnan(d)] = deletevalue
            darray = d.astype(np.float32)

            dfs.WriteItemTimeStep(item + 1, timestep, time, darray)

    dfs_i.Close()
    dfs.Close()


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
    infilename_a = str(infilename_a)
    infilename_b = str(infilename_b)
    outfilename = str(outfilename)
    copyfile(infilename_a, outfilename)

    dfs_i_a = DfsFileFactory.DfsGenericOpen(infilename_a)
    dfs_i_b = DfsFileFactory.DfsGenericOpen(infilename_b)
    dfs_o = DfsFileFactory.DfsGenericOpenEdit(outfilename)

    deletevalue = dfs_i_a.FileInfo.DeleteValueFloat

    n_time_steps = dfs_i_a.FileInfo.TimeAxis.NumberOfTimeSteps
    n_items = len(dfs_i_a.ItemInfo)
    # TODO Add checks to verify identical structure of file a and b

    for timestep in trange(n_time_steps):
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

    dfs_i_a.Close()
    dfs_i_b.Close()
    dfs_o.Close()


# TODO sum is conflicting with the built-in sum function, which we could haved used above.
def sum(
    infilename_a: str | pathlib.Path,
    infilename_b: str | pathlib.Path,
    outfilename: str | pathlib.Path,
) -> None:
    """Sum two dfs files (a+b)."""
    # deprecated
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
    """Calculate difference between two dfs files (a-b)."""
    _process_dfs_files(infilename_a, infilename_b, outfilename, operator.sub)


def concat(
    infilenames: Sequence[str | pathlib.Path],
    outfilename: str | pathlib.Path,
    keep: str = "last",
) -> None:
    """Concatenates files along the time axis.

    Overlap handling is defined by the `keep` argument,  by default the last one will be used.

    Parameters
    ----------
    infilenames:
        filenames to concatenate
    outfilename:
        filename of output
    keep:
        either 'first' (keep older), 'last' (keep newer)
        or 'average' can be selected. By default 'last'

    Notes
    ------

    The list of input files have to be sorted, i.e. in chronological order

    """
    # fast path for Dfs0
    suffix = pathlib.Path(infilenames[0]).suffix
    if suffix == ".dfs0":
        dss = [mikeio.read(f) for f in infilenames]
        ds = mikeio.Dataset.concat(dss, keep=keep)  # type: ignore
        ds.to_dfs(outfilename)
        return

    dfs_i_a = DfsFileFactory.DfsGenericOpen(str(infilenames[0]))

    dfs_o = _clone(str(infilenames[0]), str(outfilename))

    n_items = len(dfs_i_a.ItemInfo)
    dfs_i_a.Close()

    current_time = datetime(1, 1, 1)  # beginning of time...

    for i, infilename in enumerate(tqdm(infilenames, disable=not show_progress)):
        dfs_i = DfsFileFactory.DfsGenericOpen(str(infilename))
        t_axis = dfs_i.FileInfo.TimeAxis
        n_time_steps = t_axis.NumberOfTimeSteps
        dt = t_axis.TimeStep
        start_time = t_axis.StartDateTime

        if i > 0 and start_time > current_time + timedelta(seconds=dt):
            dfs_o.Close()
            os.remove(outfilename)
            raise Exception("Gap in time axis detected - not supported")

        current_time = start_time

        if keep == "last":
            if i < (len(infilenames) - 1):
                dfs_n = DfsFileFactory.DfsGenericOpen(str(infilenames[i + 1]))
                next_start_time = dfs_n.FileInfo.TimeAxis.StartDateTime
                dfs_n.Close()

            for timestep in range(n_time_steps):
                current_time = start_time + timedelta(seconds=timestep * dt)
                if i < (len(infilenames) - 1):
                    if current_time >= next_start_time:
                        break

                for item in range(n_items):
                    itemdata = dfs_i.ReadItemTimeStep(item + 1, timestep)
                    d = itemdata.Data

                    darray = d.astype(np.float32)

                    dfs_o.WriteItemTimeStepNext(0, darray)
            dfs_i.Close()

        if keep == "first":
            if (
                i == 0
            ):  # all timesteps in first file are kept (is there a more efficient way to do this without the loop?)
                for timestep in range(n_time_steps):
                    current_time = start_time + timedelta(seconds=timestep * dt)

                    for item in range(n_items):
                        itemdata = dfs_i.ReadItemTimeStep(item + 1, timestep)
                        d = itemdata.Data

                        darray = d.astype(np.float32)

                        dfs_o.WriteItemTimeStepNext(0, darray)
                end_time = (
                    start_time + timedelta(seconds=timestep * dt)
                )  # reuse last timestep since there is no EndDateTime attribute in t_axis.
                dfs_i.Close()

            else:
                # determine overlap in number of timesteps
                start_timestep = (
                    int((end_time - start_time) / timedelta(seconds=dt)) + 1
                )
                # loop only through those timesteps that are not in previous file
                for timestep in range(start_timestep, n_time_steps):
                    current_time = start_time + timedelta(seconds=timestep * dt)

                    for item in range(n_items):
                        itemdata = dfs_i.ReadItemTimeStep(item + 1, timestep)
                        d = itemdata.Data

                        darray = d.astype(np.float32)

                        dfs_o.WriteItemTimeStepNext(0, darray)
                end_time = start_time + timedelta(
                    seconds=timestep * dt
                )  # get end time from current file
                dfs_i.Close()

        elif keep == "average":
            ALPHA = 0.5  # averaging factor
            last_file = i == (len(infilenames) - 1)
            overlapping_with_next = False  # lets first asume no overlap

            # Find the start time of next file
            if not last_file:
                dfs_n = DfsFileFactory.DfsGenericOpen(str(infilenames[i + 1]))
                next_start_time = dfs_n.FileInfo.TimeAxis.StartDateTime
            else:
                next_start_time = datetime.max  # end of time ...

            if i == 0:
                timestep_n = 0  # have not read anything before

            # lets start where we left off (if last file overlapped)
            timestep = timestep_n
            while timestep < n_time_steps:
                current_time = start_time + timedelta(seconds=timestep * dt)
                if current_time >= next_start_time:  # false if last file
                    overlapping_with_next = True
                    break
                for item in range(n_items):
                    itemdata = dfs_i.ReadItemTimeStep(item + 1, timestep)
                    d = itemdata.Data
                    darray = d.astype(np.float32)
                    dfs_o.WriteItemTimeStepNext(0, darray)

                timestep += 1

            timestep_n = 0  # have not read anything from next file yet

            if not overlapping_with_next:
                dfs_n.Close()
                continue  # next file

            # Otherwhise read overlapping part
            while timestep < n_time_steps:
                for item in range(n_items):
                    itemdata_i = dfs_i.ReadItemTimeStep(item + 1, timestep)
                    itemdata_n = dfs_n.ReadItemTimeStep(item + 1, timestep_n)
                    d_i = itemdata_i.Data
                    d_n = itemdata_n.Data
                    d_av = d_i * ALPHA + d_n * (1 - ALPHA)
                    darray = d_av.astype(np.float32)
                    dfs_o.WriteItemTimeStepNext(0, (darray))
                timestep += 1
                timestep_n += 1

            # Close next file before opening it again
            dfs_n.Close()

    dfs_o.Close()


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
    dfs_i = DfsFileFactory.DfsGenericOpenEdit(str(infilename))

    is_layered_dfsu = dfs_i.ItemInfo[0].Name == "Z coordinate"

    time = _TimeInfo.parse(dfs_i.FileInfo.TimeAxis, start, end, step)
    item_numbers = _valid_item_numbers(
        dfs_i.ItemInfo, items, ignore_first=is_layered_dfsu
    )

    if is_layered_dfsu:
        item_numbers = [it + 1 for it in item_numbers]
        item_numbers.insert(0, 0)

    dfs_o = _clone(
        str(infilename),
        str(outfilename),
        start_time=time.file_start_new,
        timestep=time.timestep,
        items=item_numbers,
    )

    file_start_shift = 0
    if time.file_start_new is not None:
        file_start_orig = dfs_i.FileInfo.TimeAxis.StartDateTime
        file_start_shift = (time.file_start_new - file_start_orig).total_seconds()

    timestep_out = -1
    for timestep in range(time.start_step, time.end_step, step):
        for item_out, item in enumerate(item_numbers):
            itemdata = dfs_i.ReadItemTimeStep((item + 1), timestep)
            time_sec = itemdata.Time

            if time_sec > time.end_sec:
                dfs_i.Close()
                dfs_o.Close()
                return

            if time_sec >= time.start_sec:
                if item == item_numbers[0]:
                    timestep_out = timestep_out + 1
                time_sec_out = time_sec - file_start_shift

                outdata = itemdata.Data
                dfs_o.WriteItemTimeStep(
                    (item_out + 1), timestep_out, time_sec_out, outdata
                )

    dfs_i.Close()
    dfs_o.Close()


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
    dfs_i = DfsFileFactory.DfsGenericOpen(str(infilename))

    dfs_o = _clone(infilename, outfilename)

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

    for timestep in trange(1, n_time_steps, disable=not show_progress):
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

    dfs_o.Close()


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
    func = np.nanquantile if skipna else np.quantile

    dfs_i = DfsFileFactory.DfsGenericOpen(infilename)

    is_dfsu_3d = dfs_i.ItemInfo[0].Name == "Z coordinate"

    item_numbers = _valid_item_numbers(dfs_i.ItemInfo, items)

    if is_dfsu_3d and 0 in item_numbers:
        item_numbers.remove(0)  # Remove Zn item for special treatment

    n_items_in = len(item_numbers)

    n_time_steps = dfs_i.FileInfo.TimeAxis.NumberOfTimeSteps

    # TODO: better handling of different item sizes (zn...)

    ci = _ChunkInfo.from_dfs(dfs_i, item_numbers, buffer_size)

    qvec: Sequence[float] = [q] if isinstance(q, float) else q
    qtxt = [f"Quantile {q!r}" for q in qvec]
    core_items = [dfs_i.ItemInfo[i] for i in item_numbers]
    items = _get_repeated_items(core_items, prefixes=qtxt)

    if is_dfsu_3d:
        items.insert(0, dfs_i.ItemInfo[0])

    dfs_o = _clone(infilename, outfilename, items=items)

    n_items_out = len(items)
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
        # the last chunk may be smaller than the rest:
        chunk_end = ci.chunk_end(e1)

        # read all data for this chunk
        for timestep in range(n_time_steps):
            item_out = 0
            for item_no in item_numbers:
                itemdata = _read_item(dfs_i, item_no, timestep)
                data_chunk = itemdata[e1:e2]
                datalist[item_out][timestep, 0:chunk_end] = data_chunk
                item_out += 1

        # calculate quantiles (for this chunk)
        item_out = 0
        for item in range(n_items_in):
            qdat = np.zeros((len(qvec), (ci.chunk_size)))
            qdat[:, :] = func(datalist[item][:, 0:chunk_end], q=qvec, axis=0)
            for j in range(len(qvec)):
                outdatalist[item_out][e1:e2] = qdat[j, :]
                item_out += 1

        e1 = e2

    if is_dfsu_3d:
        znitemdata = dfs_i.ReadItemTimeStep(1, 0)
        # TODO should this be static Z coordinates instead?
        dfs_o.WriteItemTimeStepNext(0.0, znitemdata.Data)

    for item in range(n_items_out):
        darray = outdatalist[item].astype(np.float32)
        dfs_o.WriteItemTimeStepNext(0.0, darray)

    dfs_o.Close()


def _read_item(dfs: DfsFile, item: int, timestep: int) -> np.ndarray:
    indatatime = dfs.ReadItemTimeStep(item + 1, timestepIndex=timestep)
    indata = indatatime.Data
    has_value = indata != dfs.FileInfo.DeleteValueFloat
    indata[~has_value] = np.nan

    return indata.astype(np.float64)


def _get_repeated_items(
    items_in: list[DfsDynamicItemInfo], prefixes: list[str]
) -> list[ItemInfo]:
    item_numbers = _valid_item_numbers(items_in)
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
    dfs_out = _clone(infilename, outfilename, datatype=datatype)
    dfs_in = DfsFileFactory.DfsGenericOpen(infilename)

    # Copy dynamic item data
    sourceData = dfs_in.ReadItemTimeStepNext()
    while sourceData:
        dfs_out.WriteItemTimeStepNext(sourceData.Time, sourceData.Data)
        sourceData = dfs_in.ReadItemTimeStepNext()

    dfs_out.Close()
    dfs_in.Close()


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

    """
    dfs_i = DfsFileFactory.DfsGenericOpen(str(infilename))

    item_numbers = _valid_item_numbers(dfs_i.ItemInfo)
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

    dfs = _clone(
        str(infilename),
        str(outfilename),
        items=items,
    )

    n_time_steps = dfs_i.FileInfo.TimeAxis.NumberOfTimeSteps

    for timestep in range(n_time_steps):
        data = {}
        for item in range(n_items):
            name = dfs_i.ItemInfo[item].Name
            data[name] = dfs_i.ReadItemTimeStep(item_numbers[item] + 1, timestep).Data

        for item in items:
            func = funcs.get(item.name, None)
            if func is None:
                darray = data[item.name]
            else:
                try:
                    darray = func(data)
                except KeyError as e:
                    missing_key = e.args[0]
                    keys = ", ".join(data.keys())
                    dfs.Close()
                    pathlib.Path(outfilename).unlink()
                    raise KeyError(
                        f"Item '{missing_key}' is not available in the file. Available items: {keys}"
                    )
            dfs.WriteItemTimeStepNext(0.0, darray)

    dfs_i.Close()
    dfs.Close()
