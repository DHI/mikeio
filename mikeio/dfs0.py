import os
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from DHI.Generic.MikeZero import eumQuantity
from DHI.Generic.MikeZero.DFS import (
    DfsFileFactory,
    DfsFactory,
    DfsBuilder,
    DfsSimpleType,
    DataValueType,
    StatType,
)
from DHI.Generic.MikeZero.DFS.dfs0 import Dfs0Util

from .custom_exceptions import ItemNumbersError, InvalidDataType
from .dotnet import to_dotnet_array, to_dotnet_datetime, from_dotnet_datetime
from .dfsutil import _valid_item_numbers, _get_item_info
from .dataset import Dataset
from .eum import TimeStepUnit, EUMType, EUMUnit, ItemInfo, TimeAxisType
from .helpers import safe_length
from .base import TimeSeries


class Dfs0(TimeSeries):

    _start_time = None
    _n_items = None
    _dt = None
    _is_equidistant = None
    _title = None
    _items = None

    def __init__(self, filename=None):
        """Create a Dfs0 object for reading, writing

        Parameters
        ----------
        filename: str, optional
            File name including full path to the dfs0 file.
        """
        self._filename = filename

        if filename:
            self._read_header()

    def __repr__(self):
        out = ["<mikeio.Dfs0>"]

        if self._filename:
            out.append(f"Timeaxis: {str(self._timeaxistype)}")

        if self._n_items is not None:
            if self._n_items < 10:
                out.append("Items:")
                for i, item in enumerate(self.items):
                    out.append(f"  {i}:  {item}")
            else:
                out.append(f"Number of items: {self._n_items}")

        return str.join("\n", out)

    def _read_header(self):
        if not os.path.exists(self._filename):
            raise FileNotFoundError(self._filename)

        dfs = DfsFileFactory.DfsGenericOpen(self._filename)
        self._source = dfs
        self._deletevalue = dfs.FileInfo.DeleteValueFloat

        # Read items
        self._n_items = safe_length(dfs.ItemInfo)
        self._items = _get_item_info(dfs.ItemInfo, list(range(self._n_items)))

        self._timeaxistype = TimeAxisType(dfs.FileInfo.TimeAxis.TimeAxisType)

        if self._timeaxistype in [
            TimeAxisType.EquidistantCalendar,
            TimeAxisType.NonEquidistantCalendar,
        ]:
            self._start_time = from_dotnet_datetime(dfs.FileInfo.TimeAxis.StartDateTime)
        else:  # relative time axis
            self._start_time = datetime(1970, 1, 1)

        dfs.Close()

    def read(self, items=None, time_steps=None):
        """
        Read data from a dfs0 file.

        Parameters
        ----------
        items: list[int] or list[str], optional
            Read only selected items, by number (0-based), or by name
        time_steps: str, int or list[int], optional
            Read only selected time_steps

        Returns
        -------
        Dataset
            A dataset with data dimensions [t]
        """

        if not os.path.exists(self._filename):
            raise FileNotFoundError(f"File {self._filename} not found.")

        dfs = DfsFileFactory.DfsGenericOpen(self._filename)
        self._source = dfs

        self._n_items = safe_length(dfs.ItemInfo)
        item_numbers = _valid_item_numbers(dfs.ItemInfo, items)

        self._n_timesteps = dfs.FileInfo.TimeAxis.NumberOfTimeSteps

        if self._timeaxistype == TimeAxisType.NonEquidistantCalendar and isinstance(
            time_steps, str
        ):
            sel_time_step_str = time_steps
            time_steps = range(self._n_timesteps)
        else:
            sel_time_step_str = None

        dfs.Close()

        ds = self.__read(self._filename)
        ds = ds[item_numbers]
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
            ds = ds[sel]

        return ds

    def __read(self, filename):
        """
        Read all data from a dfs0 file.
        """
        self._time_column_index = 0  # First column is time (the rest is data)

        self._dfs = DfsFileFactory.DfsGenericOpen(filename)
        raw_data = Dfs0Util.ReadDfs0DataDouble(self._dfs)  # Bulk read the data
        all_data = self.__to_numpy(raw_data)

        matrix = self.__delete_to_nan(all_data[:, 1:])
        data = []
        for i in range(matrix.shape[1]):
            data.append(matrix[:, i])

        t_seconds = all_data[:, 0]
        time = pd.to_datetime(t_seconds, unit="s", origin=self.start_time)
        time = time.round(freq="ms")  # accept nothing finer than milliseconds

        items = list(self.__get_items())

        self._dfs.Close()

        return Dataset(data, time, items)

    def __to_numpy(self, raw_data):
        return np.fromiter(raw_data, np.float64).reshape(
            self._n_timesteps, self._n_items + 1
        )

    def __delete_to_nan(self, data):
        nan_indices = np.isclose(data, self._dfs.FileInfo.DeleteValueFloat, atol=1e-36)
        data[nan_indices] = np.nan
        return data

    def __get_items(self):
        for i in range(self._n_items):
            name = self._dfs.ItemInfo[i].Name
            item_type = EUMType(self._dfs.ItemInfo[i].Quantity.Item)
            unit = EUMUnit(self._dfs.ItemInfo[i].Quantity.Unit)
            value_type = self._dfs.ItemInfo[i].ValueType
            yield ItemInfo(name, item_type, unit, data_value_type=value_type)

    @staticmethod
    def _to_dfs_datatype(dtype):
        if dtype is None:
            return DfsSimpleType.Float

        if dtype in (np.float64, DfsSimpleType.Double, "double"):
            return DfsSimpleType.Double

        if dtype in (np.float32, DfsSimpleType.Float, "float", "single"):
            return DfsSimpleType.Float

        raise InvalidDataType()

    def _setup_header(self):
        factory = DfsFactory()
        builder = DfsBuilder.Create(self._title, "DFS", 0)
        builder.SetDataType(1)
        builder.SetGeographicalProjection(factory.CreateProjectionUndefined())

        system_start_time = to_dotnet_datetime(self._start_time)

        if self._is_equidistant:
            temporal_axis = factory.CreateTemporalEqCalendarAxis(
                self._timeseries_unit, system_start_time, 0, self._dt
            )
        else:
            temporal_axis = factory.CreateTemporalNonEqCalendarAxis(
                self._timeseries_unit, system_start_time
            )

        builder.SetTemporalAxis(temporal_axis)
        builder.SetItemStatisticsType(StatType.RegularStat)

        dtype_dfs = self._to_dfs_datatype(self._dtype)

        for i in range(self._n_items):
            item = self._items[i]
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
            builder.CreateFile(self._filename)
        except IOError:
            raise IOError(f"Cannot create dfs0 file: {self._filename}")

        return builder.GetFile()

    def write(
        self,
        filename,
        data,
        start_time=None,
        timeseries_unit=TimeStepUnit.SECOND,
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
        timeseries_unit: TimestepUnit, optional
            Timestep  unit, default TimeStepUnit.SECOND
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
        self._filename = filename
        self._title = title
        self._timeseries_unit = timeseries_unit
        self._dtype = dtype

        if isinstance(data, Dataset):
            self._items = data.items

            if data.is_equidistant:
                self._start_time = data.time[0]
                self._dt = (data.time[1] - data.time[0]).total_seconds()
            else:
                datetimes = data.time
            data = data.data

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
        else:
            self._is_equidistant = True
            if self._start_time is None:
                self._start_time = datetime.now()
                warnings.warn(
                    f"No start time supplied. Using current time: {self._start_time} as start time."
                )

            self._dt = np.float(self._dt)
            datetimes = np.array(
                [
                    self._start_time + timedelta(seconds=(step * self._dt))
                    for step in np.arange(self._n_timesteps)
                ]
            )

        dfs = self._setup_header()

        delete_value = dfs.FileInfo.DeleteValueFloat

        data = np.array(data).astype(np.float64)
        data[np.isnan(data)] = delete_value
        data_to_write = to_dotnet_array(data.T)
        t_seconds = [(t - datetimes[0]).total_seconds() for t in datetimes]
        Dfs0Util.WriteDfs0DataDouble(dfs, t_seconds, data_to_write)

        dfs.Close()

    def to_dataframe(self, unit_in_name=False, round_time="ms"):
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
        ds = self.read()
        df = ds.to_dataframe(unit_in_name)

        if round_time:
            rounded_idx = pd.DatetimeIndex(ds.time).round(round_time)
            df.index = pd.DatetimeIndex(rounded_idx, freq="infer")
        else:
            df.index = pd.DatetimeIndex(ds.time, freq="infer")

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

    @property
    def end_time(self):
        return self._end_time

    @property
    def n_timesteps(self):
        """Number of time steps"""
        return self._n_timesteps


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

    dfs = Dfs0()

    data = []
    for i in range(self.values.shape[1]):
        data.append(self.values[:, i])

    if items is None:

        if itemtype is None:
            items = [ItemInfo(name) for name in self.columns]
        else:
            if unit is None:
                items = [ItemInfo(name, itemtype) for name in self.columns]
            else:
                items = [ItemInfo(name, itemtype, unit) for name in self.columns]

    if self.index.freq is None:  # non-equidistant
        dfs.write(
            filename=filename,
            data=data,
            datetimes=self.index,
            items=items,
            title=title,
            dtype=dtype,
        )
    else:  # equidistant
        dt = self.index.freq.delta.total_seconds()
        start_time = self.index[0].to_pydatetime()
        dfs.write(
            filename=filename,
            data=data,
            start_time=start_time,
            dt=dt,
            items=items,
            title=title,
            dtype=dtype,
        )


# Monkey patching onto Pandas classes
pd.DataFrame.to_dfs0 = dataframe_to_dfs0

pd.Series.to_dfs0 = series_to_dfs0


def dataset_to_dfs0(self, filename):
    """Write Dataset to a Dfs0 file

    Parameters
    ----------
    filename: str
        full path and file name to the dfs0 file.
    """
    self = self.squeeze()

    if len(self.data[0].shape) != 1:
        raise ValueError(
            """Only data with a single dimension can be converted to a dfs0.
                 Hint: use `isel` to create a subset."""
        )

    dfs0 = Dfs0()

    dfs0.write(filename, self)


Dataset.to_dfs0 = dataset_to_dfs0
