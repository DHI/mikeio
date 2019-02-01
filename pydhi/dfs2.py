from pydhi import *

class dfs2:

    def __calculate_index(self, nx, ny, x, y):
        """ Calculates the position in the dfs2 data array based on the
        number of x,y  (nx,ny) at the specified x,y position.

        Error checking is done here to see if the x,y coordinates are out of range.
        """
        if x >= nx:
            raise Warning('x coordinate is off the grid: ', x)
        if y >= ny:
            raise Warning('y coordinate is off the grid: ', y)

        return y * nx + x

    def read(self, dfs2file, item_numbers):
        """ Function: Read a dfs2 file

        usage:
            [data,time,name] = read( filename, item_numbers)
            item_numbers is a list of indices (base 0) to read from

        Returns
            1) the data contained in a dfs2 file in a list of numpy matrices
            2) time index
            3) name of the items

        NOTE
            Returns data ( y, x, nt)
        """

        # NOTE. Item numbers are base 0 (everything else in the dfs is base 0)

        # Open the dfs file for reading
        dfs = DfsFileFactory.DfsGenericOpen(dfs2file);

        # Determine the size of the grid
        axis = dfs.ItemInfo.Items.get_Item(0).SpatialAxis
        yNum = axis.YCount
        xNum = axis.XCount
        nt = dfs.FileInfo.TimeAxis.NumberOfTimeSteps
        if nt == 0:
            raise Warning("Static dfs2 files (with no time steps) are not supported.")
            nt = 1
        deleteValue = dfs.FileInfo.DeleteValueFloat

        n_items = len(item_numbers)
        data_list = []

        for item in range(n_items):
            # Initialize an empty data block
            data = np.ndarray(shape=(yNum, xNum, nt), dtype=float)  # .fill(deleteValue)
            data_list.append(data)

        t = []
        startTime = dfs.FileInfo.TimeAxis.StartDateTime;
        for it in range(dfs.FileInfo.TimeAxis.NumberOfTimeSteps):
            for item in range(n_items):

                itemdata = dfs.ReadItemTimeStep(item_numbers[item] + 1, it)

                if x64:
                    src = itemdata.Data
                    src_hndl = GCHandle.Alloc(src, GCHandleType.Pinned)
                    try:
                        src_ptr = src_hndl.AddrOfPinnedObject().ToInt64()
                        bufType = ctypes.c_float * len(src)
                        cbuf = bufType.from_address(src_ptr)
                        d = np.frombuffer(cbuf, dtype=cbuf._type_)
                    finally:
                        if src_hndl.IsAllocated: src_hndl.Free()

                else:
                    raise Warning("Slow read if using 32 bit Python.")
                    d = np.array(list(itemdata.Data))

                #d = np.array(list(itemdata.Data))

                d = d.reshape(yNum, xNum)
                d = np.flipud(d)
                d[d == deleteValue] = np.nan
                data_list[item][:, :, it] = d

            t.append(startTime.AddSeconds(itemdata.Time).ToString("yyyy-MM-dd HH:mm:ss"))

        time = pd.DatetimeIndex(t)
        names = []
        for item in range(n_items):
            name = dfs.ItemInfo.Items[item_numbers[item]].Name
            names.append(name)

        dfs.Close()
        return (data_list, time, names)


    def write(self, dfs2file, data):
        """
        Function: write to a pre-created dfs2 file.

        dfs2file:
            full path and filename to existing dfs2 file

        data:
            list of matrices. len(data) must equal the number of items in the dfs2.
            Easch matrix must be of dimension y,x,time

        usage:
            write( filename, data) where  data( y, x, nt)

        Returns:
            Nothing

        """

        # Open the dfs file for writing
        dfs = DfsFileFactory.Dfs2FileOpenEdit(dfs2file)

        # Determine the size of the grid
        number_y = dfs.SpatialAxis.YCount
        number_x = dfs.SpatialAxis.XCount
        n_time_steps = dfs.FileInfo.TimeAxis.NumberOfTimeSteps
        n_items = dfs.ItemInfo.Count

        deletevalue = -1e-035

        if not all( np.shape(d)[0] == number_y for d in data):
            raise Warning("ERROR data matrices in the Y dimension do not all match in the data list. "
                     "Data is list of matices [y,x,time]")
        if not all(np.shape(d)[1] == number_x for d in data):
            raise Warning("ERROR data matrices in the X dimension do not all match in the data list. "
                     "Data is list of matices [y,x,time]")
        if not all(np.shape(d)[2] == n_time_steps for d in data):
            raise Warning("ERROR data matrices in the time dimension do not all match in the data list. "
                     "Data is list of matices [y,x,time]")
        if not len(data) == n_items:
            raise Warning("The number of matrices in data do not match the number of items in the dfs2 file.")

        for it in range(n_time_steps):
            for item in range(n_items):
                d = data[item][:, :, it]
                d[np.isnan(d)] = deletevalue
                d = d.reshape(number_y, number_x)
                d = np.flipud(d)
                darray = Array[System.Single](np.array(d.reshape(d.size, 1)[:, 0]))
                dfs.WriteItemTimeStepNext(0, darray)

        dfs.Close()

    def create_equidistant_calendar(self, dfs2file, data, start_time, timeseries_unit, dt, variable_type, unit, coordinate,
                                    x0, y0, length_x, length_y, names=None, title=None):
        """
        Creates a dfs2 file

        dfs2file:
            Location to write the dfs2 file
        data:
            list of matrices, one for each item. Matrix dimension: y, x, time
        start_time:
            start date of type datetime.
        timeseries_unit:
            second=1400, minute=1401, hour=1402, day=1403, month=1405, year= 1404
        dt:
            The time step (double based on the timeseries_unit). Therefore dt of 5.5 with timeseries_unit of minutes
            means 5 mins and 30 seconds.
        variable_type:
            Array integers corresponding to a variable types (ie. Water Level). Use dfsutil type_list
            to figure out the integer corresponding to the variable.
        unit:
            Array integers corresponding to the unit corresponding to the variable types The unit (meters, seconds),
            use dfsutil unit_list to figure out the corresponding unit for the variable.
        coordinate:
            ['UTM-33', 12.4387, 55.2257, 327]  for UTM, Long, Lat, North to Y orientation. Note: long, lat in decimal degrees
            OR
            [TODO: Support not Local Coordinates ...]
        x0:
            Lower right position
        x0:
            Lower right position
        length_x:
            length of each grid in the x direction (meters)
        length_y:
            length of each grid in the y direction (meters)
        names:
            array of names (ie. array of strings). (can be blank)
        title:
            title of the dfs2 file (can be blank)

        """

        if title is None:
            title = "dfs0 file"

        number_y = np.shape(data[0])[0]
        number_x = np.shape(data[0])[1]
        n_time_steps = np.shape(data[0])[2]
        n_items = len(data)

        if not all( np.shape(d)[0] == number_y for d in data):
            raise Warning("ERROR data matrices in the Y dimension do not all match in the data list. "
                     "Data is list of matices [y,x,time]")
        if not all(np.shape(d)[1] == number_x for d in data):
            raise Warning("ERROR data matrices in the X dimension do not all match in the data list. "
                     "Data is list of matices [y,x,time]")
        if not all(np.shape(d)[2] == n_time_steps for d in data):
            raise Warning("ERROR data matrices in the time dimension do not all match in the data list. "
                     "Data is list of matices [y,x,time]")

        if len(names) is not n_items:
            raise Warning("names must be an array of strings with the same number as matrices in data list")

        if len(variable_type) is not n_items or not all(isinstance(item, int) and 0 <= item < 1e15 for item in variable_type):
            raise Warning("type if specified must be an array of integers (enuType) with the same number of "
                          "elements as data columns")

        if len(unit) is not n_items or not all(isinstance(item, int) and 0 <= item < 1e15 for item in unit):
            raise Warning(
                "unit if specified must be an array of integers (enuType) with the same number of "
                "elements as data columns")

        if not type(start_time) is datetime.datetime:
            raise Warning("start_time must be of type datetime ")

        if not isinstance(timeseries_unit, int):
            raise Warning("timeseries_unit must be an integer. timeseries_unit: second=1400, minute=1401, hour=1402, "
                          "day=1403, month=1405, year= 1404See dfsutil options for help ")

        system_start_time = System.DateTime(start_time.year, start_time.month, start_time.day,
                                            start_time.hour, start_time.minute, start_time.second)

        # Create an empty dfs2 file object
        factory = DfsFactory();
        builder = Dfs2Builder.Create(title, 'Matlab DFS', 0);

        # Set up the header
        builder.SetDataType(1);
        builder.SetGeographicalProjection(factory.CreateProjectionGeoOrigin(coordinate[0], coordinate[1], coordinate[2], coordinate[3]))
        builder.SetTemporalAxis(
            factory.CreateTemporalEqCalendarAxis(timeseries_unit, system_start_time, 0, dt))
        builder.SetSpatialAxis(factory.CreateAxisEqD2(eumUnit.eumUmeter, number_x, x0, length_x, number_y, y0, length_y))


        for i in range(n_items):
            builder.AddDynamicItem(names[i], eumQuantity.Create(variable_type[i], unit[i]), DfsSimpleType.Float, DataValueType.Instantaneous)

        try:
            builder.CreateFile(dfs2file)
        except IOError:
            print('cannot create dfs2 file: ', dfs2file)

        dfs = builder.GetFile();
        deletevalue = dfs.FileInfo.DeleteValueFloat #-1.0000000031710769e-30

        for i in range(n_time_steps):
            for item in range(n_items):
                d = data[item][:, :, i]
                d[np.isnan(d)] = deletevalue
                d = d.reshape(number_y, number_x)
                d = np.flipud(d)
                darray = Array[System.Single](np.array(d.reshape(d.size, 1)[:, 0]))
                dfs.WriteItemTimeStepNext(0, darray)

        dfs.Close()
