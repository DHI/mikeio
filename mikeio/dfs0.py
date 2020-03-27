import os
import numpy as np
from datetime import datetime
import System
from System import Array

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

from .helpers import safe_length
from .dutil import Dataset, find_item
from .eum import TimeStep, Item, Unit, ItemInfo


class Dfs0:
    def __read(self, filename):
        """Read data from the dfs0 file
        """
        if not os.path.exists(filename):
            raise Warning("filename - File does not Exist %s", filename)

        dfs = DfsFileFactory.DfsGenericOpen(filename)
        self._dfs = dfs

        n_items = safe_length(dfs.ItemInfo)
        nt = dfs.FileInfo.TimeAxis.NumberOfTimeSteps

        items = []
        for i in range(n_items):
            eumItem = dfs.ItemInfo[i].Quantity.Item
            eumUnit = dfs.ItemInfo[i].Quantity.Unit
            name = dfs.ItemInfo[i].Name
            variable = Item(eumItem)
            unit = Unit(eumUnit)
            item = ItemInfo(name, variable, unit)
            items.append(item)

        # BULK READ THE DFS0
        dfsdata = Dfs0Util.ReadDfs0DataDouble(dfs)

        t = []
        starttime = dfs.FileInfo.TimeAxis.StartDateTime

        # EMPTY Data Block for copying the Results
        for it in range(dfs.FileInfo.TimeAxis.NumberOfTimeSteps):
            t.append(
                starttime.AddSeconds(dfsdata[it, 0]).ToString("yyyy-MM-dd HH:mm:ss")
            )

        # Copies the System Array to a numpy matrix
        # First column in the time (the rest is the data)
        data = np.fromiter(dfsdata, np.float64).reshape(nt, n_items + 1)[:, 1::]

        data[data == -1.0000000180025095e-35] = np.nan
        data[data == -1.0000000031710769e-30] = np.nan
        data[data == dfs.FileInfo.DeleteValueFloat] = np.nan
        data[data == dfs.FileInfo.DeleteValueDouble] = np.nan

        dfs.Close()

        return data, t, items

    def read_to_pandas(self, filename, item_numbers=None):
        """Read data from the dfs0 file and return a Pandas DataFrame
        Usage:
            read_to_pandas(filename, item_numbers=None)
        filename
            full path and file name to the dfs0 file.
        item_numbers
            read only the item_numbers in the array specified (0 base)

        Return:
            a Pandas DataFrame
        """
        import pandas as pd

        if item_numbers is not None:
            if not all(
                isinstance(item, int) and 0 <= item < 1e15 for item in item_numbers
            ):
                raise Warning("item_numbers must be a list of integers")

        data, t, names = self.__read(filename=filename)

        df = pd.DataFrame(data, columns=names)

        df.index = pd.DatetimeIndex(t)

        if item_numbers is not None:
            df = df.iloc[:, item_numbers]

        return df

    def read(self, filename, item_numbers=None, item_names=None):
        """Read data from the dfs0 file

        Usage:
            read(filename, item_numbers=None, item_names=None)
        filename
            full path and file name to the dfs0 file.
        item_numbers
            read only the item_numbers in the array specified (0 base)
        item_names
            read only the items in the array specified, (takes precedence over item_numbers)

        Return:
            Dataset(data, time, items)
        """

        d, t, items = self.__read(filename)

        if item_names is not None:
            item_numbers = find_item(self._dfs, item_names)

        if item_numbers is not None:
            if not all(
                isinstance(item, int) and 0 <= item < 1e15 for item in item_numbers
            ):
                raise Warning(
                    "item_numbers must be a list or array of values between 0 and 1e15"
                )

        t = [datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in t]

        data = []

        if item_numbers is not None:
            sel_items = []
            for item in item_numbers:
                data.append(d[:, item])
                sel_items.append(items[item])
            items = sel_items
        else:
            for item in range(d.shape[1]):
                data.append(d[:, item])

        return Dataset(data, t, items)

    def write(self, filename, data):

        """write overwrites an existing dfs0 file.

        filename:
            Full path and filename to dfs0 to be modified.
        data:
            a list of numpy array
        """

        if not os.path.exists(filename):
            raise Warning("filename - File does not Exist %s", filename)

        try:
            dfs = DfsFileFactory.DfsGenericOpenEdit(filename)
        except IOError:
            print("cannot open", filename)

        delete_value = dfs.FileInfo.DeleteValueFloat

        n_items = len(dfs.ItemInfo)
        n_time_steps = dfs.FileInfo.TimeAxis.NumberOfTimeSteps

        # Makes sure the data to write to the file matches the dfs0 file
        if n_time_steps != data[0].shape[0]:
            raise Exception(
                f"Inconsistent data size. nt (row count) must be size {n_time_steps}"
            )

        if n_items != len(data):
            raise Exception(f"Number of items must be size {n_items}")

        for i in range(n_items):
            d = data[i]

            d[np.isnan(d)] = delete_value

        d = Array.CreateInstance(System.Single, 1)

        # Get the date times in seconds (from start)
        dfsdata = Dfs0Util.ReadDfs0DataDouble(dfs)

        t = []
        # starttime = dfs.FileInfo.TimeAxis.StartDateTime

        for it in range(dfs.FileInfo.TimeAxis.NumberOfTimeSteps):
            t.append(dfsdata[it, 0])

        dfs.Reset()

        # COPY OVER THE DATA
        for it in range(n_time_steps):

            for ii in range(n_items):

                d = Array[System.Single](np.array(data[ii][it : it + 1]))

                # dt = (t[it] - t[0]).total_seconds()
                dt = t[it]
                dfs.WriteItemTimeStepNext(dt, d)

        dfs.Close()

    def create(
        self,
        filename,
        data,
        start_time=None,
        timeseries_unit=TimeStep.SECOND,
        dt=1,
        datetimes=None,
        items=None,
        title=None,
        data_value_type=None,
    ):
        """Create a dfs0 file.

        Parameters
        ----------
        filename: str
            Full path and filename to dfs0 to be created.
        data: list[np.array]
            values
        start_time: datetime.dateime, , optional
            start date of type datetime.
        timeseries_unit: Timestep, optional
            Timestep  unitdefault Timestep.SECOND
        dt: float, optional
            the time step. Therefore dt of 5.5 with timeseries_unit of minutes
            means 5 mins and 30 seconds. default to 1
        items: list[ItemInfo], optional
            List of ItemInfo corresponding to a variable types (ie. Water Level).
        title: str, optional
            title
        data_value_type: list[DataValueType], optional
            DataValueType default DataValueType.INSTANTANEOUS

        """
        if title is None:
            title = "dfs0 file"

        n_items = len(data)
        n_time_steps = np.shape(data[0])[0]

        if start_time is None:
            start_time = datetime.now()

        if items is None:
            items = [ItemInfo(f"temItem {i+1}") for i in range(n_items)]

        if len(items) != n_items:
            raise Warning(
                "names must be an array of strings with the same number of elements as data columns"
            )

        if datetimes is None:
            equidistant = True

            if not type(start_time) is datetime:
                raise Warning("start_time must be of type datetime ")
        else:
            start_time = datetimes[0]
            equidistant = False

        # if not isinstance(timeseries_unit, int):
        #    raise Warning("timeseries_unit must be an integer. See dfsutil options for help ")

        system_start_time = System.DateTime(
            start_time.year,
            start_time.month,
            start_time.day,
            start_time.hour,
            start_time.minute,
            start_time.second,
        )

        factory = DfsFactory()
        builder = DfsBuilder.Create(title, "DFS", 0)
        builder.SetDataType(1)
        builder.SetGeographicalProjection(factory.CreateProjectionUndefined())

        if equidistant:
            builder.SetTemporalAxis(
                factory.CreateTemporalEqCalendarAxis(
                    timeseries_unit, system_start_time, 0, dt
                )
            )
        else:
            builder.SetTemporalAxis(
                factory.CreateTemporalNonEqCalendarAxis(
                    timeseries_unit, system_start_time
                )
            )

        builder.SetItemStatisticsType(StatType.RegularStat)

        for i in range(n_items):

            item = builder.CreateDynamicItemBuilder()

            item.Set(
                items[i].name,
                eumQuantity.Create(items[i].item, items[i].unit),
                DfsSimpleType.Float,
            )

            if data_value_type is not None:
                item.SetValueType(data_value_type[i])
            else:
                item.SetValueType(DataValueType.Instantaneous)

            item.SetAxis(factory.CreateAxisEqD0())
            builder.AddDynamicItem(item.GetDynamicItemInfo())

        try:
            builder.CreateFile(filename)

        except IOError:
            print("cannot create dfso file: ", filename)

        dfs = builder.GetFile()
        delete_value = dfs.FileInfo.DeleteValueFloat

        for i in range(n_items):
            d = data[i]

            d[np.isnan(d)] = delete_value

        # COPY OVER THE DATA
        for it in range(n_time_steps):
            for ii in range(n_items):

                d = Array[System.Single](np.array(data[ii][it : it + 1]))
                if equidistant:
                    dfs.WriteItemTimeStepNext(it, d)
                else:
                    dt = (datetimes[it] - datetimes[0]).total_seconds()
                    dfs.WriteItemTimeStepNext(dt, d)

        dfs.Close()
