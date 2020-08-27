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
from .dutil import Dataset, find_item, get_valid_items_and_timesteps
from .eum import TimeStep, EUMType, EUMUnit, ItemInfo
from .helpers import safe_length


class Dfs0:

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
        

        Returns
        -------
            Dataset(data, time, items)
        """

        if not os.path.exists(self._filename):
            raise FileNotFoundError(f"File {self._filename} not found.")

        dfs = DfsFileFactory.DfsGenericOpen(self._filename)
        self._source = dfs
        self._n_items = safe_length(dfs.ItemInfo)
        self._n_timesteps = dfs.FileInfo.TimeAxis.NumberOfTimeSteps

        items, item_numbers, time_steps = get_valid_items_and_timesteps(
            self, items, time_steps
        )

        dfs.Close()

        ds = self.__read(self._filename)
        ds = ds[item_numbers]

        return ds

    def __read(self, filename):
        """
        Read all data from a dfs0 file.
        """
        
        
        #self._n_items = safe_length(self._dfs.ItemInfo)
        #self._n_timesteps = self._dfs.FileInfo.TimeAxis.NumberOfTimeSteps
        self._time_column_index = 0  # First column is time (the rest is data)
        
        self._dfs = DfsFileFactory.DfsGenericOpen(filename)
        raw_data = Dfs0Util.ReadDfs0DataDouble(self._dfs)  # Bulk read the data
        
        matrix = self.__to_numpy_with_nans(raw_data)

        data = []
        for i in range(matrix.shape[1]):
            data.append(matrix[:,i])

        time = list(self.__get_time(raw_data))
        items = list(self.__get_items())

        self._dfs.Close()

        

        return Dataset(data, time, items)

    def __to_numpy_with_nans(self, raw_data):
        data = np.fromiter(raw_data, np.float64).reshape(self._n_timesteps, self._n_items + 1)[:, 1::]
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
        if not all(isinstance(item_number, int) and 0 <= item_number < 1e15 for item_number in item_numbers):
            raise Warning("item_numbers must be a list or array of values between 0 and 1e15")

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
      

    def overwrite(self, filename, data):
        """
        Overwrite data in an existing dfs0 file.

        Parameters
        ----------
        filename: str
            Full path and filename to dfs0 to be modified.
        data: list[np.array]
            New data to write.
        """
        warnings.warn("This method is deprecated, use write instead")

        dfs, n_items, n_time_steps = self._validate_and_open_dfs(filename, data)

        # Get time in seconds from start
        existing_data = Dfs0Util.ReadDfs0DataDouble(dfs)
        time = [existing_data[i, 0] for i in range(n_time_steps)]

        # Overwrite with new data
        dfs.Reset()
        new_data = np.nan_to_num(data, nan=dfs.FileInfo.DeleteValueFloat)
        new_data_dotnet = to_dotnet_array(np.stack(new_data, axis=1))
        Dfs0Util.WriteDfs0DataDouble(dfs, time, new_data_dotnet)
        dfs.Close()

    @staticmethod
    def _to_dfs_datatype(dtype):
        if dtype is None:
            return DfsSimpleType.Float

        if dtype in (np.float64, DfsSimpleType.Double, "double"):
            return DfsSimpleType.Double

        if dtype in (np.float32, DfsSimpleType.Float, "float", "single"):
            return DfsSimpleType.Float

        raise ValueError("Invalid data type. Choose np.float32 or np.float64")

    def write(
            self,
            filename,
            data,
            start_time=None,
            timeseries_unit=TimeStep.SECOND,
            dt=1.0,
            datetimes=None,
            items=None,
            title=None,
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
            title
        data_value_type: list[DataValueType], optional
            DataValueType default DataValueType.INSTANTANEOUS
        dtype : np.dtype, optional
            default np.float32

        """

        if isinstance(data, Dataset):
            items = data.items
            start_time = data.time[0]
            if dt is None and len(data.time) > 1:
                if not data.is_equidistant:
                    raise Exception(
                        "Data is not equidistant in time. Dfsu requires equidistant temporal axis!"
                    )
                dt = (data.time[1] - data.time[0]).total_seconds()
            data = data.data

        n_items = len(data)
        n_time_steps = np.shape(data[0])[0]

        if start_time is None:
            start_time = datetime.now()

        if items is None:
            items = [ItemInfo(f"Item {i + 1}") for i in range(n_items)]

        if len(items) != n_items:
            raise Warning(
                "names must be an array of strings with the same number of elements as data columns"
            )

        if datetimes is None:
            equidistant = True

            #if not type(start_time) is datetime:
            #    raise Warning("start_time must be of type datetime.")

            dt = np.float(dt)
            datetimes = np.array(
                [
                    start_time + timedelta(seconds=(step * dt))
                    for step in np.arange(n_time_steps)
                ]
            )

        else:
            start_time = datetimes[0]
            equidistant = False

        system_start_time = to_dotnet_datetime(start_time)

        if title is None:
            title = "dfs0 file"

        factory = DfsFactory()
        builder = DfsBuilder.Create(title, "DFS", 0)
        builder.SetDataType(1)
        builder.SetGeographicalProjection(factory.CreateProjectionUndefined())

        if equidistant:
            temporal_axis = factory.CreateTemporalEqCalendarAxis(timeseries_unit, system_start_time, 0, dt)
        else:
            temporal_axis = factory.CreateTemporalNonEqCalendarAxis(timeseries_unit, system_start_time)

        builder.SetTemporalAxis(temporal_axis)
        builder.SetItemStatisticsType(StatType.RegularStat)

        dtype_dfs = self._to_dfs_datatype(dtype)

        for i in range(n_items):
            item = builder.CreateDynamicItemBuilder()
            quantity = eumQuantity.Create(items[i].type, items[i].unit)
            item.Set(items[i].name, quantity, dtype_dfs, )

            if data_value_type is not None:
                item.SetValueType(data_value_type[i])
            else:
                item.SetValueType(DataValueType.Instantaneous)

            item.SetAxis(factory.CreateAxisEqD0())
            builder.AddDynamicItem(item.GetDynamicItemInfo())

        try:
            builder.CreateFile(filename)
        except IOError:
            raise IOError(f"Cannot create dfs0 file: {filename}")

        dfs = builder.GetFile()

        delete_value = dfs.FileInfo.DeleteValueFloat

        data = data.copy()
        for i in range(n_items):
            d = data[i].copy()
            d[np.isnan(d)] = delete_value
            

        data_to_write = to_dotnet_array(np.stack(data, axis=1))
        t_seconds = [(t - datetimes[0]).total_seconds() for t in datetimes]
        Dfs0Util.WriteDfs0DataDouble(dfs, t_seconds, data_to_write)

        dfs.Close()

    def to_dataframe(self, unit_in_name=False, round_time='s'):
        """
        Read data from the dfs0 file and return a Pandas DataFrame.
        
        Parameters
        ----------
        filename: str
            full path and file name to the dfs0 file.
        unit_in_name: bool, optional
            include unit in column name, default False
        round_time: string, bool, optional
            round time to avoid problem with floating point inaccurcy, set to False to avoid rounding
        Returns
        -------
        pd.DataFrame
        """
        ds = self.read()
        df = ds.to_dataframe()

        if round_time:
            rounded_idx = pd.DatetimeIndex(ds.time).round(round_time)
            df.index = pd.DatetimeIndex(rounded_idx, freq="infer")
        else:
            df.index = pd.DatetimeIndex(t, freq="infer")

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
