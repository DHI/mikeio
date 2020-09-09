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

from .dotnet import to_dotnet_array, to_dotnet_datetime, from_dotnet_datetime
from .dutil import Dataset, get_valid_items_and_timesteps
from .eum import TimeStep, EUMType, EUMUnit, ItemInfo
from .helpers import safe_length


class Dfs0:

    _start_time = None
    _n_items = None
    _dt = None
    _is_equidistant = None
    _title = None
    _data_value_type = None
    _items = None

    def __init__(self, filename=None):
        """Create a Dfs0 object for reading, writing

        Parameters
        ----------
        filename: str, optional
            File name including full path to the dfs0 file.
        """
        self._filename = filename

    def read(self, items=None, time_steps=None):
        """
        Read data from a dfs0 file.

        Parameters
        ----------
        items: list[int] or list[str], optional
            Read only selected items, by number (0-based), or by name
        time_steps: int or list[int], optional
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
        self._n_timesteps = dfs.FileInfo.TimeAxis.NumberOfTimeSteps

        items, item_numbers, _ = get_valid_items_and_timesteps(self, items, time_steps)

        dfs.Close()

        ds = self.__read(self._filename)
        ds = ds[item_numbers]
        if time_steps:
            ds = ds.isel(time_steps, axis=0)

        return ds

    def __read(self, filename):
        """
        Read all data from a dfs0 file.
        """
        self._time_column_index = 0  # First column is time (the rest is data)

        self._dfs = DfsFileFactory.DfsGenericOpen(filename)
        raw_data = Dfs0Util.ReadDfs0DataDouble(self._dfs)  # Bulk read the data

        matrix = self.__to_numpy_with_nans(raw_data)

        data = []
        for i in range(matrix.shape[1]):
            data.append(matrix[:, i])

        time = list(self.__get_time(raw_data))
        items = list(self.__get_items())

        self._dfs.Close()

        return Dataset(data, time, items)

    def __to_numpy_with_nans(self, raw_data):
        data = np.fromiter(raw_data, np.float64).reshape(
            self._n_timesteps, self._n_items + 1
        )[:, 1::]
        nan_indices = np.isclose(data, self._dfs.FileInfo.DeleteValueFloat, atol=1e-36)
        data[nan_indices] = np.nan
        return data

    def __get_time(self, raw_data):
        start_time = from_dotnet_datetime(self._dfs.FileInfo.TimeAxis.StartDateTime)

        for t in range(self._n_timesteps):
            t_sec = raw_data[t, self._time_column_index]
            yield start_time + timedelta(seconds=t_sec)

    def __get_items(self):
        for i in range(self._n_items):
            name = self._dfs.ItemInfo[i].Name
            item_type = EUMType(self._dfs.ItemInfo[i].Quantity.Item)
            unit = EUMUnit(self._dfs.ItemInfo[i].Quantity.Unit)
            yield ItemInfo(name, item_type, unit)

    @staticmethod
    def _validate_item_numbers(item_numbers):
        if not all(
            isinstance(item_number, int) and 0 <= item_number < 1e15
            for item_number in item_numbers
        ):
            raise Warning(
                "item_numbers must be a list or array of values between 0 and 1e15"
            )

    @staticmethod
    def _validate_and_open_dfs(filename, data):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found.")

        try:
            dfs = DfsFileFactory.DfsGenericOpenEdit(filename)
        except IOError:
            raise IOError(f"Cannot open {filename}.")

        n_items = len(dfs.ItemInfo)
        n_time_steps = dfs.FileInfo.TimeAxis.NumberOfTimeSteps

        # Match the data to write to the existing dfs0 file
        if n_time_steps != data[0].shape[0]:
            raise Exception(
                f"Inconsistent data size. Number of time steps (row count) is {data[0].shape[0]}. Expected {n_time_steps}."
            )

        if n_items != len(data):
            raise Exception(f"The number of items is {len(data)}. Expected {n_items}.")

        return dfs, n_items, n_time_steps

    @staticmethod
    def _to_dfs_datatype(dtype):
        if dtype is None:
            return DfsSimpleType.Float

        if dtype in (np.float64, DfsSimpleType.Double, "double"):
            return DfsSimpleType.Double

        if dtype in (np.float32, DfsSimpleType.Float, "float", "single"):
            return DfsSimpleType.Float

        raise ValueError("Invalid data type. Choose np.float32 or np.float64")

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
                item.name, quantity, dtype_dfs,
            )

            if self._data_value_type is not None:
                newitem.SetValueType(self._data_value_type[i])
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
        timeseries_unit=TimeStep.SECOND,
        dt=None,
        datetimes=None,
        items=None,
        title="",
        data_value_type=None,
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
        timeseries_unit: Timestep, optional
            Timestep  unitdefault Timestep.SECOND
        dt: float, optional
            the time step. Therefore dt of 5.5 with timeseries_unit of minutes
            means 5 mins and 30 seconds. default to 1.0
        datetimes: list[datetime]
        items: list[ItemInfo], optional
            List of ItemInfo corresponding to a variable types (ie. Water Level).
        title: str, optional
            title, default blank
        data_value_type: list[DataValueType], optional
            DataValueType default DataValueType.INSTANTANEOUS
        dtype : np.dtype, optional
            default np.float32

        """
        self._filename = filename
        self._title = title
        self._timeseries_unit = timeseries_unit
        self._dtype = dtype
        self._data_value_type = data_value_type

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
        self._n_time_steps = np.shape(data[0])[0]

        if items:
            self._items = items

        if self._items is None:
            warnings.warn("No items info supplied. Using Item 1, 2, 3,...")
            self._items = [ItemInfo(f"Item {i + 1}") for i in range(self._n_items)]

        if len(self._items) != self._n_items:
            raise ValueError(
                "names must be an array of strings with the same number of elements as data columns"
            )

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
                    for step in np.arange(self._n_time_steps)
                ]
            )

        dfs = self._setup_header()

        delete_value = dfs.FileInfo.DeleteValueFloat

        data = np.array(data)
        data[np.isnan(data)] = delete_value
        data_to_write = to_dotnet_array(data.T)
        t_seconds = [(t - datetimes[0]).total_seconds() for t in datetimes]
        Dfs0Util.WriteDfs0DataDouble(dfs, t_seconds, data_to_write)

        dfs.Close()

    def to_dataframe(self, unit_in_name=False, round_time="s"):
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


def dataframe_to_dfs0(
    self,
    filename,
    itemtype=None,
    unit=None,
    items=None,
    title=None,
    data_value_type=None,
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
    data_value_type: list[DataValueType], optional
            DataValueType default DataValueType.INSTANTANEOUS
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
            data_value_type=data_value_type,
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
            data_value_type=data_value_type,
            dtype=dtype,
        )


pd.DataFrame.to_dfs0 = dataframe_to_dfs0
