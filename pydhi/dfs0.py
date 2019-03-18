from pydhi import *

class dfs0:

    def __read(self, dfs0file):
        """Read data from the dfs0 file
        """
        if not path.exists(dfs0file):
            raise Warning("dfs0File - File does not Exist %s", dfs0file)

        dfs = DfsFileFactory.DfsGenericOpen(dfs0file);

        n_items = len(dfs.ItemInfo)
        nt = dfs.FileInfo.TimeAxis.NumberOfTimeSteps

        names = []
        for i in range(n_items):
            names.append(dfs.ItemInfo[i].Name)

        # BULK READ THE DFS0
        dfsdata = Dfs0Util.ReadDfs0DataDouble(dfs)

        t = []
        starttime = dfs.FileInfo.TimeAxis.StartDateTime;

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



    def read_to_pandas(self, dfs0file, indices=None):
        """Read data from the dfs0 file and return a Pandas DataFrame
        Usage:
            read_to_pandas(dfs0file, indices=None)
        dfs0file
            full path and file name to the dfs0 file.
        indices
            read only the indices in the array specified (0 base)

        Return:
            a Pandas DataFrame
        """
        import pandas as pd
        from operator import itemgetter

        if indices is not None:
            if not all(isinstance(item, int) and 0 <= item < 1e15 for item in indices):
                raise Warning("indices must be a list or array of values between 0 and 1e15")

        data, t, names = self.__read(dfs0file=dfs0file)

        if indices is not None:
            data = data[:, indices]
            names = itemgetter(*indices)(names)

        df = pd.DataFrame(data, columns=names)

        df.index = pd.DatetimeIndex(t)

        return df


    def read(self, dfs0file, indices=None):
        """Read data from the dfs0 file and return data [data, time, itemNames]

        Usage:
            read_to_pandas(dfs0file, indices=None)
        dfs0file
            full path and file name to the dfs0 file.
        indices
            read only the indices in the array specified (0 base)
        Return:
            [data, time, itemNames]
        """
        from operator import itemgetter

        if indices is not None:
            if not all(isinstance(item, int) and 0 <= item < 1e15 for item in indices):
                raise Warning("indices must be a list or array of values between 0 and 1e15")

        data, t, names = self.__read(dfs0file)

        if indices is not None:
            data = data[:, indices]
            names = itemgetter(*indices)(names)

        return data, t, names


    def write(self, dfs0file, data):
        """Writes data to the pre-created dfs0 file.
        dfs0file --> file path to existing dfs0 file.
        data --> numpy matrix with data.
        """

        if not path.exists(dfs0file):
            raise Warning("dfs0File - File does not Exist %s", dfs0file)

        try:
            dfs = DfsFileFactory.DfsGenericOpenEdit(dfs0file);
        except IOError:
            print('cannot open', dfs0file)

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


    def create_equidistant_calendar(self, dfs0file, data, start_time, timeseries_unit, dt, variable_type, unit, names=None,
                                    title=None, data_value_type=None):
        """Create_equidistant_calendar creates a dfs0 file with Equidistant Calendar.

        dfs0file:
            Full path and filename to dfs0 to be created.
        data:
            a numpy matrix
        start_time:
            start date of type datetime.
        timeseries_unit:
            second=1400, minute=1401, hour=1402, day=1403, month=1405, year= 1404
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

        n_items = np.shape(data)[1]
        n_time_steps = np.shape(data)[0]

        if names is not None and len(names) is not n_items:
            raise Warning("names must be an array of strings with the same number of elements as data columns")

        if len(variable_type) is not n_items :
            raise Warning("type if specified must be an array of integers (enuType) with the same number of "
                          "elements as data columns")

        if len(unit) is not n_items:
            raise Warning(
                "unit if specified must be an array of integers (enuType) with the same number of "
                "elements as data columns")

        if not type(start_time) is datetime.datetime:
            raise Warning("start_time must be of type datetime ")

        if not isinstance(timeseries_unit, int):
            raise Warning("timeseries_unit must be an integer. See dfsutil options for help ")

        system_start_time = System.DateTime(start_time.year, start_time.month, start_time.day,
                                            start_time.hour, start_time.minute, start_time.second)

        factory = DfsFactory()
        builder = DfsBuilder.Create(title, 'DFS', 0);
        builder.SetDataType(1);
        builder.SetGeographicalProjection(factory.CreateProjectionUndefined());
        builder.SetTemporalAxis(factory.CreateTemporalEqCalendarAxis(timeseries_unit, system_start_time, 0, dt))
        builder.SetItemStatisticsType(StatType.RegularStat);

        for i in range(n_items):

            item = builder.CreateDynamicItemBuilder()
            if type is not None:
                item.Set(names[i], eumQuantity.Create(variable_type[i], unit[i]), DfsSimpleType.Float)
            else:
                item.Set(str(i), eumQuantity.Create(eumItem.eumIItemUndefined, 0), DfsSimpleType.Float)

            if data_value_type is not None:
                item.SetValueType(data_value_type[i])
            else:
                item.SetValueType(DataValueType.Instantaneous)

            item.SetAxis(factory.CreateAxisEqD0());
            builder.AddDynamicItem(item.GetDynamicItemInfo());

        try:
            builder.CreateFile(dfs0file)

        except IOError:
            print('cannot create dfso file: ', dfs0file)

        dfs = builder.GetFile()
        delete_value = dfs.FileInfo.DeleteValueFloat

        data[np.isnan(data)] = delete_value

        # COPY OVER THE DATA
        for it in range(n_time_steps):
            for ii in range(n_items):
                d = Array[System.Single](np.array([[data[it, ii]]]))
                dfs.WriteItemTimeStepNext(it, d)

        dfs.Close()



    def create_non_equidistant_calendar(self, dfs0file, data, time_vector, variable_type, unit, names=None,
                                    title=None, data_value_type=None):
        """Create_non_equidistant_calendar creates a dfs0 file with NOT-Equidistant Calendar.

        dfs0file: Full path and filename to dfs0 to be created.
        data: a numpy matrix
        time_vector: A list of datetime elements.
        variable_type: Array integers corresponding to a variable types (ie. Water Level). Use dfsutil type_list
            to figure out the integer corresponding to the variable.
        unit: Array integers corresponding to the unit corresponding to the variable types The unit (meters, seconds),
            use dfsutil unit_list to figure out the corresponding unit for the variable.
        names: array of names (ie. array of strings)
        title: title (string)
        data_value_type:  Instantaneous = 0 (default), Accumulated = 1, StepAccumulated = 3, MeanStepBackward = 3,
            MeanStepForward = 4.

        """

        if title is None:
            title = "dfs0 file"

        n_items = np.shape(data)[1]
        n_time_steps = np.shape(data)[0]

        if names is not None and len(names) is not n_items:
            raise Warning("names must be an array of strings with the same number of elements as data columns")

        if len(variable_type) is not n_items or not all(isinstance(item, int) and 0 <= item < 1e15
                                                        for item in variable_type):
            raise Warning("type if specified must be an array of integers (enuType) with the same number of "
                          "elements as data columns")

        if not len(time_vector) == n_time_steps or not all(isinstance(t, datetime.datetime)for t in time_vector):
            raise Warning("The time_vector must be an array or list of datetime of same length as the number of "
                          "rows in the data")

        if len(unit) is not n_items or not all(isinstance(item, int) and 0 <= item < 1e15 for item in unit):
            raise Warning(
                "unit if specified must be an array of integers (enuType) with the same number of "
                "elements as data columns")

        start_time = time_vector[0]
        system_start_time = System.DateTime(start_time.year, start_time.month, start_time.day,
                                            start_time.hour, start_time.minute, start_time.second)

        # default eumUnit --> second = 1400
        timeseries_unit = 1400

        factory = DfsFactory()
        builder = DfsBuilder.Create(title, 'DFS', 0);
        builder.SetDataType(1);
        builder.SetGeographicalProjection(factory.CreateProjectionUndefined());
        builder.SetTemporalAxis(factory.CreateTemporalNonEqCalendarAxis(timeseries_unit, system_start_time))
        builder.SetItemStatisticsType(StatType.RegularStat);

        for i in range(n_items):

            item = builder.CreateDynamicItemBuilder()
            if type is not None:
                item.Set(names[i], eumQuantity.Create(variable_type[i], unit[i]), DfsSimpleType.Float)
            else:
                item.Set(str(i), eumQuantity.Create(eumItem.eumIItemUndefined, 0), DfsSimpleType.Float)

            if data_value_type is not None:
                item.SetValueType(data_value_type[i])
            else:
                item.SetValueType(DataValueType.Instantaneous)

            item.SetAxis(factory.CreateAxisEqD0());
            builder.AddDynamicItem(item.GetDynamicItemInfo());

        try:
            builder.CreateFile(dfs0file)
        except IOError:
            print('cannot create dfso file: ', dfs0file)

        dfs = builder.GetFile()
        delete_value = dfs.FileInfo.DeleteValueFloat

        data[np.isnan(data)] = delete_value


        # COPY OVER THE DATA
        for it in range(n_time_steps):
            dt = (time_vector[it] - time_vector[0]).total_seconds()
            for ii in range(n_items):
                d = Array[System.Single](np.array([[data[it, ii]]]))
                dfs.WriteItemTimeStepNext(dt, d)

        dfs.Close()

