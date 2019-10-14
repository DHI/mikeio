import os
import numpy as np
from datetime import datetime
import System
from System import Array
from DHI.Generic.MikeZero import eumQuantity, eumItem
from DHI.Generic.MikeZero.DFS import DfsFileFactory, DfsFactory, DfsBuilder, DfsSimpleType, DataValueType, StatType
from DHI.Generic.MikeZero.DFS.dfs0 import Dfs0Util

from pydhi.helpers import safe_length
from pydhi.dutil import Dataset


class dfs0():

    def __read(self, filename):
        """Read data from the dfs0 file
        """
        if not os.path.exists(filename):
            raise Warning("filename - File does not Exist %s", filename)

        dfs = DfsFileFactory.DfsGenericOpen(filename)

        n_items = safe_length(dfs.ItemInfo)
        nt = dfs.FileInfo.TimeAxis.NumberOfTimeSteps

        names = []
        for i in range(n_items):
            names.append(dfs.ItemInfo[i].Name)

        # BULK READ THE DFS0
        dfsdata = Dfs0Util.ReadDfs0DataDouble(dfs)

        t = []
        starttime = dfs.FileInfo.TimeAxis.StartDateTime

        # EMPTY Data Block for copying the Results
        for it in range(dfs.FileInfo.TimeAxis.NumberOfTimeSteps):
            t.append(starttime.AddSeconds(dfsdata[it, 0]).ToString("yyyy-MM-dd HH:mm:ss"))

        # Copies the System Array to a numpy matrix
        # First column in the time (the rest is the data)
        data = np.fromiter(dfsdata, np.float64).reshape(nt, n_items + 1)[:, 1::]

        data[data == -1.0000000180025095e-35] = np.nan
        data[data == -1.0000000031710769e-30] = np.nan
        data[data == dfs.FileInfo.DeleteValueFloat] = np.nan
        data[data == dfs.FileInfo.DeleteValueDouble] = np.nan

        dfs.Close()

        return data, t, names

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
            if not all(isinstance(item, int) and 0 <= item < 1e15 for item in item_numbers):
                raise Warning("item_numbers must be a list of integers")

        data, t, names = self.__read(filename=filename)

        df = pd.DataFrame(data, columns=names)

        df.index = pd.DatetimeIndex(t)

        if item_numbers is not None:
            df = df.iloc[:,item_numbers]

        return df

    def read(self, filename, item_numbers=None):
        """Read data from the dfs0 file and return data [data, time, itemNames]

        Usage:
            read_to_pandas(filename, item_numbers=None)
        filename
            full path and file name to the dfs0 file.
        item_numbers
            read only the item_numbers in the array specified (0 base)
        Return:
            [data, time, itemNames]
        """
        from operator import itemgetter

        if item_numbers is not None:
            if not all(isinstance(item, int) and 0 <= item < 1e15 for item in item_numbers):
                raise Warning("item_numbers must be a list or array of values between 0 and 1e15")

        d, t, names = self.__read(filename)

        data = []

        if item_numbers is not None:
            names = itemgetter(*item_numbers)(names)
            for item in item_numbers:
                data.append(d[:,item])
        else:
            for item in range(d.shape[1]):
                data.append(d[:,item])

        return Dataset(data, t, names)

    def write(self, filename, data):
        """Writes data to the pre-created dfs0 file.
        filename --> file path to existing dfs0 file.
        data --> numpy matrix with data.
        """

        if not path.exists(filename):
            raise Warning("filename - File does not Exist %s", filename)

        try:
            dfs = DfsFileFactory.DfsGenericOpenEdit(filename)
        except IOError:
            print('cannot open', filename)

        delete_value = dfs.FileInfo.DeleteValueFloat

        n_items = len(dfs.ItemInfo)
        nt = dfs.FileInfo.TimeAxis.NumberOfTimeSteps

        if len(np.shape(data)) == 1:
            data = data.reshape(len(data), 1)

        # Makes sure the data to write to the file matches the dfs0 file
        if nt != data.shape[0]:
            print("Inconsistent data size. nt (row count) must be size" + str(nt))
            # quit()
        if n_items != data.shape[1]:
            print("Inconsistent data size. number of items (column count) must be size" + str(n_items))

        data[np.isnan(data)] = delete_value

        d = Array.CreateInstance(System.Single, 1)

        # Get the date times in seconds (from start)
        t = []
        for i in range(nt):
            itemData = dfs.ReadItemTimeStep(1, i)
            newTime = DfsExtensions.TimeInSeconds(itemData, dfs.FileInfo.TimeAxis)
            t.append(newTime)

        dfs.Reset()

        # COPY OVER THE DATA
        for it in range(dfs.FileInfo.TimeAxis.NumberOfTimeSteps):

            #itemData = dfs.ReadItemTimeStep(1, it)
            #newTime = DfsExtensions.TimeInSeconds(itemData, dfs.FileInfo.TimeAxis)
            #tit = System.Double(newTime)
            tit = System.Double(t[it])
            for ii in range(len(dfs.ItemInfo)):
                d = Array[System.Single](np.array([[data[it, ii]]]))
                dfs.WriteItemTimeStepNext(tit, d)

        dfs.Close()

    def create(self, filename, data,
               start_time=None, timeseries_unit=1400, dt=3600, datetimes=None,
               variable_type=None, unit=None, names=None,
               title=None, data_value_type=None):
        """create creates a dfs0 file.

        filename:
            Full path and filename to dfs0 to be created.
        data:
            a numpy matrix
        start_time:
            start date of type datetime.
        timeseries_unit:
            second=1400, minute=1401, hour=1402, day=1403, month=1405, year= 1404, default=1400
        dt:
            the time step (double based on the timeseries_unit). Therefore dt of 5.5 with timeseries_unit of minutes
            means 5 mins and 30 seconds.
        variable_type:
            Array integers corresponding to a variable types (ie. Water Level). Use dfsutil type_list
            to figure out the integer corresponding to the variable.
        unit:
            Array integers corresponding to the unit corresponding to the variable types The unit (meters, seconds),
            use dfsutil unit_list to figure out the corresponding unit for the variable.
        names:
            array of names (ie. array of strings)
        title:
            title (string)
        data_value_type:
            Instantaneous = 0 (default), Accumulated = 1, StepAccumulated = 3, MeanStepBackward = 3,
            MeanStepForward = 4.

        """
        if title is None:
            title = "dfs0 file"

        n_items = len(data)
        n_time_steps = np.shape(data[0])[0]

        if start_time is None:
            start_time = datetime.now()

        if names is None:
            names = [f"Item {i+1}" for i in range(n_items)]

        if variable_type is None:
            variable_type = [999] * n_items

        if unit is None:
            unit = [0] * n_items

        if names is not None and len(names) != n_items:
            raise Warning(
                "names must be an array of strings with the same number of elements as data columns")

        if len(variable_type) != n_items:
            raise Warning("type if specified must be an array of integers (eumType) with the same number of "
                          "elements as data columns")

        if len(unit) != n_items:
            raise Warning(
                "unit if specified must be an array of integers (eumType) with the same number of "
                "elements as data columns")

        if datetimes is None:
            equidistant = True

            if not type(start_time) is datetime:
                raise Warning("start_time must be of type datetime ")
        else:
            start_time = datetimes[0]
            equidistant = False

        if not isinstance(timeseries_unit, int):
            raise Warning("timeseries_unit must be an integer. See dfsutil options for help ")

        system_start_time = System.DateTime(start_time.year, start_time.month, start_time.day,
                                            start_time.hour, start_time.minute, start_time.second)

        factory = DfsFactory()
        builder = DfsBuilder.Create(title, 'DFS', 0)
        builder.SetDataType(1)
        builder.SetGeographicalProjection(factory.CreateProjectionUndefined())

        if equidistant:
            builder.SetTemporalAxis(factory.CreateTemporalEqCalendarAxis(
                timeseries_unit, system_start_time, 0, dt))
        else:
            builder.SetTemporalAxis(factory.CreateTemporalNonEqCalendarAxis(
                timeseries_unit, system_start_time))

        builder.SetItemStatisticsType(StatType.RegularStat)

        for i in range(n_items):

            item = builder.CreateDynamicItemBuilder()
            if type is not None:
                item.Set(names[i], eumQuantity.Create(
                    variable_type[i], unit[i]), DfsSimpleType.Float)
            else:
                item.Set(str(i), eumQuantity.Create(
                    eumItem.eumIItemUndefined, 0), DfsSimpleType.Float)

            if data_value_type is not None:
                item.SetValueType(data_value_type[i])
            else:
                item.SetValueType(DataValueType.Instantaneous)

            item.SetAxis(factory.CreateAxisEqD0())
            builder.AddDynamicItem(item.GetDynamicItemInfo())

        try:
            builder.CreateFile(filename)

        except IOError:
            print('cannot create dfso file: ', filename)

        dfs = builder.GetFile()
        delete_value = dfs.FileInfo.DeleteValueFloat

        for i in range(n_items):
            d = data[i]

            d[np.isnan(d)] = delete_value

        # COPY OVER THE DATA
        for it in range(n_time_steps):
            for ii in range(n_items):

                d = Array[System.Single](np.array(data[ii][it:it+1]))
                if equidistant:
                    dfs.WriteItemTimeStepNext(it, d)
                else:
                    dt = (datetimes[it] - datetimes[0]).total_seconds()
                    dfs.WriteItemTimeStepNext(dt, d)

        dfs.Close()
