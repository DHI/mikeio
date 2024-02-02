from __future__ import annotations
from functools import cached_property
from pathlib import Path
import warnings
from datetime import datetime, timedelta
from typing import Sequence

import numpy as np
import pandas as pd
from mikecore.DfsFactory import DfsBuilder, DfsFactory  # type: ignore
from mikecore.DfsFile import DataValueType, DfsSimpleType, StatType, TimeAxisType  # type: ignore
from mikecore.DfsFileFactory import DfsFileFactory  # type: ignore
from mikecore.eum import eumQuantity  # type: ignore

from .. import __dfs_version__
from ..dataset import Dataset, DataArray
from ._dfs import _get_item_info, _valid_item_numbers, _valid_timesteps
from ..eum import EUMType, EUMUnit, ItemInfo, TimeStepUnit


def _write_dfs0(filename, dataset: Dataset, title="", dtype=DfsSimpleType.Float):
    filename = str(filename)

    factory = DfsFactory()
    builder = DfsBuilder.Create(title, "mikeio", __dfs_version__)
    builder.SetDataType(1)
    builder.SetGeographicalProjection(factory.CreateProjectionUndefined())

    system_start_time = dataset.time[0]

    if dataset.is_equidistant:
        if len(dataset.time) == 1:
            dt = 1.0  # TODO
        else:
            dt = (dataset.time[1] - dataset.time[0]).total_seconds()  # type: ignore

        temporal_axis = factory.CreateTemporalEqCalendarAxis(
            TimeStepUnit.SECOND, system_start_time, 0, dt
        )
    else:
        temporal_axis = factory.CreateTemporalNonEqCalendarAxis(
            TimeStepUnit.SECOND, system_start_time
        )

    builder.SetTemporalAxis(temporal_axis)
    builder.SetItemStatisticsType(StatType.RegularStat)

    dfs_dtype = Dfs0._to_dfs_datatype(dtype)

    for da in dataset:
        newitem = builder.CreateDynamicItemBuilder()
        quantity = eumQuantity.Create(da.type, da.unit)
        newitem.Set(da.name, quantity, dfs_dtype)

        # TODO set default on DataArray
        if da.item.data_value_type is not None:
            newitem.SetValueType(da.item.data_value_type)
        else:
            newitem.SetValueType(DataValueType.Instantaneous)

        newitem.SetAxis(factory.CreateAxisEqD0())
        builder.AddDynamicItem(newitem.GetDynamicItemInfo())

    builder.CreateFile(filename)

    dfs = builder.GetFile()

    delete_value = dfs.FileInfo.DeleteValueFloat

    t_seconds = (dataset.time - dataset.time[0]).total_seconds().values  # type: ignore

    data = np.array(dataset.to_numpy(), order="F").astype(np.float64).T
    data[np.isnan(data)] = delete_value

    if data.ndim == 2:
        data_to_write = np.concatenate([t_seconds.reshape(-1, 1), data], axis=1)
    else:
        data_to_write = np.concatenate(
            [np.atleast_2d(t_seconds), np.atleast_2d(data)], axis=1
        )
    dfs.WriteDfs0DataDouble(data_to_write)

    dfs.Close()


class Dfs0:
    def __init__(self, filename: str | Path):
        """Create a Dfs0 object for reading, writing

        Parameters
        ----------
        filename: str or Path
            File name including full path to the dfs0 file.
        """
        self._filename = str(filename)

        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(path)

        dfs = DfsFileFactory.DfsGenericOpen(str(path))
        self._source = dfs

        # Read items
        self._n_items = len(dfs.ItemInfo)
        self._items = _get_item_info(dfs.ItemInfo, list(range(self._n_items)))

        self._timeaxistype = dfs.FileInfo.TimeAxis.TimeAxisType

        if self._timeaxistype in {
            TimeAxisType.CalendarEquidistant,
            TimeAxisType.CalendarNonEquidistant,
        }:
            self._start_time = dfs.FileInfo.TimeAxis.StartDateTime
        else:  # relative time axis
            self._start_time = datetime(1970, 1, 1)

        # time
        self._n_timesteps = dfs.FileInfo.TimeAxis.NumberOfTimeSteps

        dfs.Close()

    def __repr__(self):
        out = ["<mikeio.Dfs0>"]
        out.append(f"timeaxis: {repr(self._timeaxistype)}")

        if self._n_items < 10:
            out.append("items:")
            for i, item in enumerate(self.items):
                out.append(f"  {i}:  {item}")
        else:
            out.append(f"number of items: {self._n_items}")

        return str.join("\n", out)

    def read(self, items=None, time=None, keepdims=False) -> Dataset:
        """
        Read data from a dfs0 file.

        Parameters
        ----------
        items: list[int] or list[str], optional
            Read only selected items, by number (0-based), or by name
        time: int, str, datetime, pd.TimeStamp, sequence, slice or pd.DatetimeIndex, optional
            Read only selected time steps, by default None (=all)

        Returns
        -------
        Dataset
            A Dataset with data dimensions [t]
        """
        path = Path(self._filename)
        if not path.exists():
            raise FileNotFoundError(f"File {path} not found")

        # read data from file
        fdata, ftime, fitems = self.__read(self._filename)
        self._source = self._dfs
        dfs = self._dfs

        # select items
        self._n_items = len(dfs.ItemInfo)
        item_numbers = _valid_item_numbers(dfs.ItemInfo, items)
        if items is not None:
            fdata = [fdata[it] for it in item_numbers]
            fitems = [fitems[it] for it in item_numbers]
        ds = Dataset(fdata, ftime, fitems, validate=False)

        # select time steps
        self._n_timesteps = dfs.FileInfo.TimeAxis.NumberOfTimeSteps
        if self._timeaxistype == TimeAxisType.CalendarNonEquidistant and isinstance(
            time, str
        ):
            sel_time_step_str = time
            time_steps = None
        else:
            sel_time_step_str = None
            time_steps = None
            if time is not None:
                _, time_steps = _valid_timesteps(dfs.FileInfo, time)

        if time_steps:
            ds = ds.isel(time_steps, axis=0)

        if sel_time_step_str:
            parts = sel_time_step_str.split(",")
            if len(parts) == 1:
                parts.append(parts[0])  # end=start

            if parts[0] == "":
                sel = slice(parts[1])  # stop only
            elif parts[1] == "":
                sel = slice(parts[0], None)  # start only
            else:
                sel = slice(parts[0], parts[1])
            ds = ds[sel]  # type: ignore

        return ds

    def __read(self, filename):
        """
        Read all data from a dfs0 file.
        """
        self._dfs = DfsFileFactory.DfsGenericOpen(filename)
        raw_data = self._dfs.ReadDfs0DataDouble()  # Bulk read the data

        self._dfs.Close()

        matrix = raw_data[:, 1:]
        # matrix[matrix == self._deletevalue] = np.nan
        matrix[matrix == self._dfs.FileInfo.DeleteValueDouble] = np.nan  # cutil
        matrix[matrix == self._dfs.FileInfo.DeleteValueFloat] = np.nan  # linux
        data = []
        for i in range(matrix.shape[1]):
            data.append(matrix[:, i])

        t_seconds = raw_data[:, 0]
        time = pd.to_datetime(t_seconds, unit="s", origin=self.start_time)
        time = time.round(freq="ms")  # accept nothing finer than milliseconds

        items = [
            ItemInfo(
                item.Name,
                EUMType(item.Quantity.Item),
                EUMUnit(item.Quantity.Unit),
                data_value_type=item.ValueType,
            )
            for item in self._dfs.ItemInfo
        ]

        return data, time, items

    @staticmethod
    def _to_dfs_datatype(dtype):
        if dtype is None:
            return DfsSimpleType.Float

        if dtype in {np.float64, DfsSimpleType.Double, "double"}:
            return DfsSimpleType.Double

        if dtype in {np.float32, DfsSimpleType.Float, "float", "single"}:
            return DfsSimpleType.Float

        raise TypeError("Dfs files only support float or double")

    @staticmethod
    def _setup_header(
        title: str,
        filename: str,
        start_time,
        dt: float,
        is_equidistant: bool,
        dtype,
        items: Sequence[ItemInfo],
    ):
        factory = DfsFactory()
        builder = DfsBuilder.Create(title, "mikeio", __dfs_version__)
        builder.SetDataType(1)
        builder.SetGeographicalProjection(factory.CreateProjectionUndefined())

        if is_equidistant:
            temporal_axis = factory.CreateTemporalEqCalendarAxis(
                TimeStepUnit.SECOND, start_time, 0, dt
            )
        else:
            temporal_axis = factory.CreateTemporalNonEqCalendarAxis(
                TimeStepUnit.SECOND, start_time
            )

        builder.SetTemporalAxis(temporal_axis)
        builder.SetItemStatisticsType(StatType.RegularStat)

        dtype_dfs = Dfs0._to_dfs_datatype(dtype)

        for item in items:
            newitem = builder.CreateDynamicItemBuilder()
            quantity = eumQuantity.Create(item.type, item.unit)
            newitem.Set(
                item.name,
                quantity,
                dtype_dfs,
            )

            if item.data_value_type is not None:
                newitem.SetValueType(item.data_value_type)
            else:
                newitem.SetValueType(DataValueType.Instantaneous)

            newitem.SetAxis(factory.CreateAxisEqD0())
            builder.AddDynamicItem(newitem.GetDynamicItemInfo())

        try:
            builder.CreateFile(filename)
        except IOError:
            raise IOError(f"Cannot create dfs0 file: {filename}")

        return builder.GetFile()

    def write(
        self,
        filename,
        data,
        start_time=None,
        dt=None,
        datetimes=None,
        items=None,
        title="",
        dtype=None,
    ):
        """
        Create a dfs0 file.

        Parameters
        ----------
        filename: str
            Full path and filename to dfs0 to be created.
        data: list[np.array]
            values
        start_time: datetime.datetime, , optional
            start date of type datetime.
        dt: float, optional
            the time step. Therefore dt of 5.5 with timeseries_unit of minutes
            means 5 mins and 30 seconds. default to 1.0
        datetimes: list[datetime]
        items: list[ItemInfo], optional
            List of ItemInfo corresponding to a variable types (ie. Water Level).
        title: str, optional
            title, default blank
        dtype : np.dtype, optional
            default np.float32

        """
        self._filename = str(filename)
        self._title = title
        self._dtype = dtype

        if isinstance(data, Dataset):
            self._items = data.items

            if data.is_equidistant:
                self._start_time = data.time[0]
                self._dt = (data.time[1] - data.time[0]).total_seconds()
            else:
                datetimes = data.time
            data = data.to_numpy()
        elif datetimes is not None:
            datetimes = pd.DatetimeIndex(datetimes, freq="infer")

        if dt:
            self._dt = dt

        if self._dt is None:
            self._dt = 1.0

        if start_time:
            self._start_time = start_time

        self._n_items = len(data)
        self._n_timesteps = np.shape(data[0])[0]

        if items:
            self._items = items

        if self._items is None:
            warnings.warn("No items info supplied. Using Item 1, 2, 3,...")
            self._items = [ItemInfo(f"Item {i + 1}") for i in range(self._n_items)]

        if len(self._items) != self._n_items:
            raise ValueError("Number of items must match the number of data columns.")

        if datetimes is not None:
            self._start_time = datetimes[0]
            self._is_equidistant = False
            t_seconds = (datetimes - datetimes[0]).total_seconds().values
        else:
            self._is_equidistant = True
            if self._start_time is None:
                self._start_time = datetime.now()
                warnings.warn(
                    f"No start time supplied. Using current time: {self._start_time} as start time."
                )

            self._dt = float(self._dt)
            t_seconds = self._dt * np.arange(float(self._n_timesteps))

        dfs = self._setup_header(
            title=self._title,
            filename=self._filename,
            dt=self._dt,
            start_time=self._start_time,
            is_equidistant=self._is_equidistant,
            dtype=self._dtype,
            items=self._items,
        )

        delete_value = dfs.FileInfo.DeleteValueFloat

        data = np.array(data, order="F").astype(np.float64).T
        data[np.isnan(data)] = delete_value
        data_to_write = np.concatenate([t_seconds.reshape(-1, 1), data], axis=1)
        rc = dfs.WriteDfs0DataDouble(data_to_write)
        if rc:
            warnings.warn(
                f"mikecore WriteDfs0DataDouble returned {rc}! Writing file probably failed."
            )

        dfs.Close()

    def to_dataframe(self, unit_in_name=False, round_time="ms") -> pd.DataFrame:
        """
        Read data from the dfs0 file and return a Pandas DataFrame.

        Parameters
        ----------
        unit_in_name: bool, optional
            include unit in column name, default False
        round_time: string, bool, optional
            round time to avoid problem with floating point inaccurcy, set to False to avoid rounding
        Returns
        -------
        pd.DataFrame
        """
        data, time, items = self.__read(self._filename)
        if unit_in_name:
            cols = [f"{item.name} ({item.unit.name})" for item in items]
        else:
            cols = [f"{item.name}" for item in items]
        df = pd.DataFrame(np.atleast_2d(data).T, index=time, columns=cols)

        if round_time:
            rounded_idx = pd.DatetimeIndex(time).round(round_time)
            df.index = pd.DatetimeIndex(rounded_idx, freq="infer")
        else:
            df.index = pd.DatetimeIndex(time, freq="infer")

        return df

    @staticmethod
    def from_dataframe(df, filename, itemtype=None, unit=None, items=None):
        """
        Create a dfs0 from a pandas Dataframe

        Parameters
        ----------

        df: pd.DataFrame
            Dataframe with data
        filename: str
            filename to write output
        itemtype: EUMType, optional
            Same type for all items
        unit: EUMUnit, optional
            Same unit for all items
        items: list[ItemInfo]
            Different types, units for each items, similar to `create`
        """
        return dataframe_to_dfs0(df, filename, itemtype, unit, items)

    @property
    def deletevalue(self):
        """File delete value"""
        return self._deletevalue

    @property
    def n_items(self):
        """Number of items"""
        return self._n_items

    @property
    def items(self):
        """List of items"""
        return self._items

    @property
    def start_time(self):
        """File start time"""
        return self._start_time

    @cached_property
    def end_time(self):

        if self._source.FileInfo.TimeAxis.IsEquidistant():
            dt = self._source.FileInfo.TimeAxis.TimeStep
            n_steps = self._source.FileInfo.TimeAxis.NumberOfTimeSteps
            timespan = dt * (n_steps - 1)
        else:
            timespan = self._source.FileInfo.TimeAxis.TimeSpan

        return self.start_time + timedelta(seconds=timespan)

    @property
    def n_timesteps(self):
        """Number of time steps"""
        return self._n_timesteps

    @property
    def timestep(self):
        """Time step size in seconds"""
        if self._timeaxistype == TimeAxisType.CalendarEquidistant:
            return self._source.FileInfo.TimeAxis.TimeStep

    @property
    def time(self):
        """File all datetimes"""
        if self._timeaxistype == TimeAxisType.CalendarEquidistant:
            return pd.to_datetime(
                [
                    self.start_time + timedelta(seconds=i * self.timestep)
                    for i in range(self.n_timesteps)
                ]
            )

        elif self._timeaxistype == TimeAxisType.CalendarNonEquidistant:
            return self.read().time

        else:
            return None


def series_to_dfs0(
    self,
    filename,
    itemtype=None,
    unit=None,
    items=None,
    title=None,
    dtype=None,
):

    df = pd.DataFrame(self)
    df.to_dfs0(filename, itemtype, unit, items, title, dtype)


def dataframe_to_dfs0(
    self,
    filename,
    itemtype=None,
    unit=None,
    items=None,
    title=None,
    dtype=None,
):
    """
    Create a dfs0

    Parameters
    ----------
    filename: str
        filename to write output
    itemtype: EUMType, optional
        Same type for all items
    unit: EUMUnit, optional
        Same unit for all items
    items: list[ItemInfo]
        Different types, units for each items, similar to `create`
    title: str, optional
        Title of dfs0 file
    dtype : np.dtype, optional
            default np.float32
    """
    if not isinstance(self.index, pd.DatetimeIndex):
        raise ValueError(
            "Dataframe index must be a DatetimeIndex. Hint: pd.read_csv(..., parse_dates=True)"
        )

    ncol = self.values.shape[1]
    data = [self.values[:, i] for i in range(ncol)]

    if items is None:
        if itemtype is None:
            items = [ItemInfo(name) for name in self.columns]
        else:
            if unit is None:
                items = [ItemInfo(name, itemtype) for name in self.columns]
            else:
                items = [ItemInfo(name, itemtype, unit) for name in self.columns]

    das = {
        item.name: DataArray(data=d, item=item, time=self.index)
        for d, item in zip(data, items)
    }
    ds = Dataset(das)
    _write_dfs0(filename=filename, dataset=ds, title=title, dtype=dtype)


# Monkey patching onto Pandas classes
pd.DataFrame.to_dfs0 = dataframe_to_dfs0  # type: ignore

pd.Series.to_dfs0 = series_to_dfs0  # type: ignore
