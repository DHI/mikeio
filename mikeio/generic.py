import os
from typing import Iterable, List, Union
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
from shutil import copyfile
from copy import deepcopy

from tqdm import trange, tqdm

from mikecore.DfsFileFactory import DfsFileFactory
from mikecore.DfsBuilder import DfsBuilder
from mikecore.DfsFile import DfsDynamicItemInfo, DfsFile
from mikecore.eum import eumQuantity
from .dfsutil import _valid_item_numbers, _get_item_info
from .eum import EUMType, ItemInfo

show_progress = False


class _ChunkInfo:
    """Class for keeping track of an chunked processing"""

    def __init__(self, n_data: int, n_chunks: np.int32):

        self.n_data = n_data
        self.n_chunks = n_chunks

    def __repr__(self):
        return f"_ChunkInfo(n_chunks={self.n_chunks}, n_data={self.n_data}, chunk_size={self.chunk_size})"

    @property
    def chunk_size(self):
        return math.ceil(self.n_data / self.n_chunks)

    def stop(self, start: int) -> int:
        return min(start + self.chunk_size, self.n_data)

    def chunk_end(self, start):
        e2 = self.stop(start)
        return self.chunk_size - ((start + self.chunk_size) - e2)

    @staticmethod
    def from_dfs(
        dfs: DfsFile, item_numbers: List[int], buffer_size: float
    ) -> "_ChunkInfo":
        """Calculate chunk info based on # of elements in dfs file and selected buffer size"""

        n_time_steps = dfs.FileInfo.TimeAxis.NumberOfTimeSteps
        n_data_all = np.sum([dfs.ItemInfo[i].ElementCount for i in item_numbers])
        mem_need = 8 * n_time_steps * n_data_all  # n_items *
        n_chunks = math.ceil(mem_need / buffer_size)
        n_data = n_data_all // len(item_numbers)

        return _ChunkInfo(n_data, n_chunks)


def _clone(infilename: str, outfilename: str, start_time=None, items=None) -> DfsFile:
    """Clone a dfs file

    Parameters
    ----------
    infilename : str
        input filename
    outfilename : str
        output filename
    start_time : datetime, optional
        new start time for the new file, default
    items : list(int,str,eum.ItemInfo), optional
        list of items for new file, either as a list of
        ItemInfo or a list of str/int referring to original file,
        default: all items from original file

    Returns
    -------
    DfsFile
        MIKE generic dfs file object
    """
    source = DfsFileFactory.DfsGenericOpen(infilename)
    fi = source.FileInfo

    builder = DfsBuilder.Create(
        fi.FileTitle, fi.ApplicationTitle, fi.ApplicationVersion
    )

    # Set up the header
    builder.SetDataType(fi.DataType)
    builder.SetGeographicalProjection(fi.Projection)

    # Copy time axis
    time_axis = fi.TimeAxis
    if start_time is not None:
        time_axis.StartDateTime = start_time
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

    names = [x.Name for x in source.ItemInfo]
    item_lookup = {name: i for i, name in enumerate(names)}

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
            elif isinstance(item, str):
                item_no = item_lookup[item]
                builder.AddDynamicItem(source.ItemInfo[item_no])

    elif isinstance(items, (int, str)) or items is None:
        # must be str/int refering to original file (or None)
        item_numbers = _valid_item_numbers(source.ItemInfo, items)
        items = [source.ItemInfo[item] for item in item_numbers]
        for item in items:
            builder.AddDynamicItem(item)
    else:
        raise ValueError("Items of type: {type(items)} is not supported")

    # Create file
    builder.CreateFile(outfilename)

    # Copy static items
    while True:
        static_item = source.ReadStaticItemNext()
        if static_item is None:
            break
        builder.AddStaticItem(static_item)

    # Get the file
    file = builder.GetFile()

    source.Close()

    return file


def scale(
    infilename: str,
    outfilename: str,
    offset: float = 0.0,
    factor: float = 1.0,
    items: Union[List[str], List[int]] = None,
) -> None:
    """Apply scaling to any dfs file

    Parameters
    ----------

    infilename: str
        full path to the input file
    outfilename: str
        full path to the output file
    offset: float, optional
        value to add to all items, default 0.0
    factor: float, optional
        value to multiply to all items, default 1.0
    items: List[str] or List[int], optional
        Process only selected items, by number (0-based) or name, by default: all
    """
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


def sum(infilename_a: str, infilename_b: str, outfilename: str) -> None:
    """Sum two dfs files (a+b)

    Parameters
    ----------
    infilename_a: str
        full path to the first input file
    infilename_b: str
        full path to the second input file
    outfilename: str
        full path to the output file
    """
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
            d_a[d_a == deletevalue] = np.nan
            time = itemdata_a.Time

            outdata = d_a + d_b

            darray = outdata.astype(np.float32)

            dfs_o.WriteItemTimeStep(item + 1, timestep, time, darray)

    dfs_i_a.Close()
    dfs_i_b.Close()
    dfs_o.Close()


def diff(infilename_a: str, infilename_b: str, outfilename: str) -> None:
    """Calculate difference between two dfs files (a-b)

    Parameters
    ----------
    infilename_a: str
        full path to the first input file
    infilename_b: str
        full path to the second input file
    outfilename: str
        full path to the output file
    """

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

            outdata = d_a - d_b

            darray = outdata.astype(np.float32)

            dfs_o.WriteItemTimeStep(item + 1, timestep, time, darray)

    dfs_i_a.Close()
    dfs_i_b.Close()
    dfs_o.Close()


def concat(infilenames: List[str], outfilename: str, keep="last") -> None:
    """Concatenates files along the time axis

    If files are overlapping, the last one will be used.

    Parameters
    ----------
    infilenames: List[str]
        filenames to concatenate
    outfilename: str
        filename of output
    keep: str
        either 'first' (keep older), 'last' (keep newer)
        or 'average' can be selected. By default 'last'

    Notes
    ------

    The list of input files have to be sorted, i.e. in chronological order
    """

    dfs_i_a = DfsFileFactory.DfsGenericOpen(infilenames[0])

    dfs_o = _clone(infilenames[0], outfilename)

    n_items = len(dfs_i_a.ItemInfo)
    dfs_i_a.Close()

    current_time = datetime(1, 1, 1)  # beginning of time...

    for i, infilename in enumerate(tqdm(infilenames, disable=not show_progress)):

        dfs_i = DfsFileFactory.DfsGenericOpen(infilename)
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
                dfs_n = DfsFileFactory.DfsGenericOpen(infilenames[i + 1])
                nf = dfs_n.FileInfo.TimeAxis.StartDateTime
                next_start_time = datetime(
                    nf.year, nf.month, nf.day, nf.hour, nf.minute, nf.second
                )
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
                end_time = start_time + timedelta(
                    seconds=timestep * dt
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

    dfs_o.Close()


def extract(infilename: str, outfilename: str, start=0, end=-1, items=None) -> None:
    """Extract timesteps and/or items to a new dfs file

    Parameters
    ----------
    infilename : str
        path to input dfs file
    outfilename : str
        path to output dfs file
    start : int, float, str or datetime, optional
        start of extraction as either step, relative seconds
        or datetime/str, by default 0 (start of file)
    end : int, float, str or datetime, optional
        end of extraction as either step, relative seconds
        or datetime/str, by default -1 (end of file)
    items : int, list(int), str, list(str), optional
        items to be extracted to new file

    Examples
    --------
    >>> extract('f_in.dfs0', 'f_out.dfs0', start='2018-1-1')
    >>> extract('f_in.dfs2', 'f_out.dfs2', end=-3)
    >>> extract('f_in.dfsu', 'f_out.dfsu', start=1800.0, end=3600.0)
    >>> extract('f_in.dfsu', 'f_out.dfsu', items=[2, 0])
    >>> extract('f_in.dfsu', 'f_out.dfsu', items="Salinity")
    >>> extract('f_in.dfsu', 'f_out.dfsu', end='2018-2-1 00:00', items="Salinity")
    """
    dfs_i = DfsFileFactory.DfsGenericOpenEdit(infilename)

    is_layered_dfsu = dfs_i.ItemInfo[0].Name == "Z coordinate"

    file_start_new, start_step, start_sec, end_step, end_sec = _parse_start_end(
        dfs_i, start, end
    )
    item_numbers = _valid_item_numbers(
        dfs_i.ItemInfo, items, ignore_first=is_layered_dfsu
    )

    if is_layered_dfsu:
        item_numbers = [it + 1 for it in item_numbers]
        item_numbers.insert(0, 0)

    dfs_o = _clone(
        infilename, outfilename, start_time=file_start_new, items=item_numbers
    )

    file_start_shift = 0
    if file_start_new is not None:
        file_start_orig = dfs_i.FileInfo.TimeAxis.StartDateTime
        file_start_shift = (file_start_new - file_start_orig).total_seconds()

    timestep_out = -1
    for timestep in range(start_step, end_step):
        for item_out, item in enumerate(item_numbers):
            itemdata = dfs_i.ReadItemTimeStep((item + 1), timestep)
            time_sec = itemdata.Time

            if time_sec > end_sec:
                dfs_i.Close()
                dfs_o.Close()
                return

            if time_sec >= start_sec:
                if item == item_numbers[0]:
                    timestep_out = timestep_out + 1
                time_sec_out = time_sec - file_start_shift

                outdata = itemdata.Data
                dfs_o.WriteItemTimeStep(
                    (item_out + 1), timestep_out, time_sec_out, outdata
                )

    dfs_i.Close()
    dfs_o.Close()


def _parse_start_end(dfs_i, start, end):
    """Helper function for parsing start and end arguments"""
    n_time_steps = dfs_i.FileInfo.TimeAxis.NumberOfTimeSteps
    file_start_datetime = dfs_i.FileInfo.TimeAxis.StartDateTime
    file_start_sec = dfs_i.FileInfo.TimeAxis.StartTimeOffset
    start_sec = file_start_sec

    timespan = 0
    if dfs_i.FileInfo.TimeAxis.TimeAxisType == 3:
        timespan = dfs_i.FileInfo.TimeAxis.TimeStep * (n_time_steps - 1)
    elif dfs_i.FileInfo.TimeAxis.TimeAxisType == 4:
        timespan = dfs_i.FileInfo.TimeAxis.TimeSpan
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
        start_step = 0
        warnings.warn("start cannot be before start of file")

    if start_sec < file_start_sec:
        start_sec = file_start_sec
        warnings.warn("start cannot be before start of file")

    if (end_sec < start_sec) or (end_step < start_step):
        raise ValueError("end must be after start")

    if end_step > n_time_steps:
        end_step = n_time_steps
        warnings.warn("end cannot be after end of file")

    if end_sec > file_end_sec:
        end_sec = file_end_sec
        warnings.warn("end cannot be after end of file")

    file_start_new = None
    if dfs_i.FileInfo.TimeAxis.TimeAxisType == 3:
        dt = dfs_i.FileInfo.TimeAxis.TimeStep
        if (start_sec > file_start_sec) and (start_step == 0):
            # we can find the coresponding step
            start_step = int((start_sec - file_start_sec) / dt)
        file_start_new = file_start_datetime + timedelta(seconds=start_step * dt)
    elif dfs_i.FileInfo.TimeAxis.TimeAxisType == 4:
        if start_sec > file_start_sec:
            file_start_new = file_start_datetime + timedelta(seconds=start_sec)

    return file_start_new, start_step, start_sec, end_step, end_sec


def avg_time(infilename: str, outfilename: str, skipna=True):
    """Create a temporally averaged dfs file

    Parameters
    ----------
    infilename : str
        input filename
    outfilename : str
        output filename
    skipna : bool, optional
        exclude NaN/delete values when computing the result, default True
    """

    dfs_i = DfsFileFactory.DfsGenericOpen(infilename)

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
        outdatalist.append(indata.astype(np.float64))
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
    infilename: str, outfilename: str, q, *, items=None, skipna=True, buffer_size=1.0e9
):
    """Create temporal quantiles of all items in dfs file

    Parameters
    ----------
    infilename : str
        input filename
    outfilename : str
        output filename
    q: array_like of float
        Quantile or sequence of quantiles to compute,
        which must be between 0 and 1 inclusive.
    items: List[str] or List[int], optional
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

    qvec = [q] if np.isscalar(q) else q
    qtxt = [f"Quantile {q}" for q in qvec]
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
    items_in: List[DfsDynamicItemInfo], prefixes: List[str]
) -> List[ItemInfo]:
    item_numbers = _valid_item_numbers(items_in)
    items_in = _get_item_info(items_in)

    new_items = []
    for item_num in item_numbers:
        for prefix in prefixes:
            item = deepcopy(items_in[item_num])
            item.name = f"{prefix}, {item.name}"
            new_items.append(item)

    return new_items
