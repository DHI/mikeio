from pydhi import *

class dfs3:


    def __calculate_index(self, nx, ny, nz, x, y, z):
        """ Calculates the position in the dfs3 data array based on the
        number of x,y,z layers (nx,ny,nz) at the specified x,y,z position.

        Error checking is done here to see if the x,y,z coordinates are out of range.
        """
        if x >= nx:
            raise Warning('x coordinate is off the grid: ', x)
        if y >= ny:
            raise Warning('y coordinate is off the grid: ', y)
        if z >= nz:
            raise Warning('z coordinate is off the grid: ', z)

        return y*nx + x + z*nx*ny


    def grid_coordinates(self, dfs3file):
        """ Function: Returns the Grid information
        Usage:
            [X0, Y0, dx, dy, nx, ny, nz, nt] = grid_coordinates( filename )
        dfs3file
            a full path and filename to the dfs3 file

        Returns:

            X0, Y0:
                bottom left coordinates
            dx, dy:
                grid size in x and y directions
            nx, ny, nz:
                number of grid elements in the x, y and z direction
            nt:
                number of time steps
        """

        dfs = DfsFileFactory.DfsGenericOpen(dfs3file);

        # Determine the size of the grid
        axis = dfs.ItemInfo.Items.get_Item(0).SpatialAxis
        dx = axis.Dx
        dy = axis.Dy
        x0 = axis.X0
        y0 = axis.Y0
        yNum = axis.YCount
        xNum = axis.XCount
        zNum = axis.ZCount
        nt = dfs.FileInfo.TimeAxis.NumberOfTimeSteps

        dfs.Close()

        return x0, y0, dx, dy, xNum, yNum, zNum, nt


    def read_slice(self, dfs3file, item_numbers, lower_left_xy, upper_right_xy, layers=None, conservative=True):
        """ Function: Read data from a dfs3 file within the locations chosen


        Usage:
            [data,time,name] = read( filename, item_numbers, lower_left_xy, upper_right_xy, conservative)
        dfs3file
            a full path and filename to the dfs3 file
        item_numbers
            list of indices (base 0) to read from
        lower_left_xy
            list or array of size two with the X and the Y coordinate (same projection as the dfs3)
        upper_right_xy
            list or array of size two with the X and the Y coordinate (same projection as the dfs3)
        layers
            list of layers to read
        conservative
            Default is true. Only include the grids within the given bounds (don't return those grids on the boarder)

        Returns
            1) the data contained in a dfs3 file in a list of numpy matrices
            2) time index
            3) name of the items

        NOTE
            Returns data ( y, x, z, nt)

            1) If coordinates is selected, then only return data at those coordinates
            2) coordinates specified overules layers.
            3) layer counts from the bottom
        """

        data = self.read(dfs3file, item_numbers, layers=layers)

        dfs = DfsFileFactory.DfsGenericOpen(dfs3file);

        # Determine the size of the grid
        axis = dfs.ItemInfo.Items.get_Item(item_numbers[0]).SpatialAxis
        dx = axis.Dx
        dy = axis.Dy
        x0 = axis.X0
        y0 = axis.Y0
        yNum = axis.YCount
        xNum = axis.XCount

        top_left_y = y0 + (yNum + 1)*dy

        dfs.Close()

        # SLICE all the Data

        lower_left_x_index = (lower_left_xy[0] - x0) / dx
        lower_left_y_index = (top_left_y - lower_left_xy[1]) / dy

        upper_right_x_index = (upper_right_xy[0] - x0) / dx
        upper_right_y_index = (top_left_y - upper_right_xy[1]) / dy

        if conservative:
            lower_left_x_index = int(np.ceil(lower_left_x_index))
            upper_right_x_index = int(np.floor(upper_right_x_index))
            lower_left_y_index = int(np.floor(lower_left_y_index))
            upper_right_y_index = int(np.ceil(upper_right_y_index))

        else:
            lower_left_x_index = int(np.floor(lower_left_x_index))
            upper_right_x_index = int(np.ceil(upper_right_x_index))
            lower_left_y_index = int(np.ceil(lower_left_y_index))
            upper_right_y_index = int(np.floor(upper_right_y_index))

        if lower_left_x_index < 0:
            raise Warning("lower_left_x_index < 0.")
            lower_left_x_index = 0

        if upper_right_y_index < 0:
            raise Warning("upper_right_y_index < 0.")
            upper_right_y_index = 0

        if lower_left_y_index > yNum - 1:
            raise Warning("lower_left_y_index > yNum - 1")
            lower_left_y_index = yNum - 1

        if upper_right_x_index > xNum - 1:
            raise Warning("upper_right_x_index > xNum - 1")
            upper_right_x_index = xNum - 1

        for i in range( len(data[0])):
            data[0][i] = data[0][i][ upper_right_y_index:lower_left_y_index,lower_left_x_index:upper_right_x_index,:,:]

        return data


    def read(self, dfs3file, item_numbers=None, layers=None, coordinates=None):
        """ Function: Read data from a dfs3 file

        Usage:
            [data,time,name] = read( filename, item_numbers, layers=None, coordinates=None)

        item_numbers
            list of indices (base 0) to read from. If none, then all.
        layers
            list of layer indices (base 0) to read
        coordinates
            list of list (x,y,layer) integers ( 0,0 at Bottom Left of Grid !! )
            example coordinates = [[2,5,1], [11,41,2]]

        Returns
            1) the data contained in a dfs3 file in a list of numpy matrices
            2) time index
            3) name of the items

        NOTE
            Returns data ( y, x, z, nt)

            1) If coordinates is selected, then only return data at those coordinates
            2) coordinates specified overules layers.
            3) layer counts from the bottom
        """

        if not x64:
            raise Warning("Not tested in 32 bit Python. It will by default use a MUCH SLOWER reader.")

        # Open the dfs file for reading
        dfs = DfsFileFactory.DfsGenericOpen(dfs3file)

        # Determine the size of the grid
        axis = dfs.ItemInfo[0].SpatialAxis
            
        zNum = axis.ZCount
        yNum = axis.YCount
        xNum = axis.XCount
        nt = dfs.FileInfo.TimeAxis.NumberOfTimeSteps

        deleteValue = dfs.FileInfo.DeleteValueFloat

        if item_numbers is None:
            item_numbers = list(range(len(dfs.ItemInfo)))

        n_items = len(item_numbers)
        data_list = []

        if coordinates is None:
            # if nt is 0, then the dfs is 'static' and must be handled differently
            if nt is not 0:
                for item in range(n_items):
                    if layers is None:
                        # Initialize an empty data block
                        data = np.ndarray(shape=(yNum, xNum, zNum, nt), dtype=float)  # .fill(deleteValue)
                        data_list.append(data)
                    else:
                        data = np.ndarray(shape=(yNum, xNum, len(layers), nt), dtype=float)  # .fill(deleteValue)
                        data_list.append(data)

            else:
                raise Warning("Static dfs3 files (with no time steps) are not supported.")
                quit()
        else:
            ncoordinates = len(coordinates)
            for item in range(n_items):
                # Initialize an empty data block
                data = np.ndarray(shape=(ncoordinates, nt), dtype=float)
                data_list.append(data)


        t = []
        startTime = dfs.FileInfo.TimeAxis.StartDateTime;

        if coordinates is None:
            for it in range(nt):
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
                        d = np.array(list(itemdata.Data))

                    # DO a direct copy instead of eleement by elment
                    d = d.reshape(zNum, yNum, xNum).swapaxes(0, 2).swapaxes(0, 1)
                    d = np.flipud(d)
                    d[d == deleteValue] = np.nan
                    if layers is None:
                        data_list[item][:, :, :, it] = d
                    else:
                        for l in range(len(layers)):
                            data_list[item][:, :, l, it] = d[:, :, layers[l]]

                t.append(startTime.AddSeconds(itemdata.Time).ToString("yyyy-MM-dd HH:mm:ss"))
        else:
            indices = [self.__calculate_index(xNum, yNum, zNum, x, y, z) for x, y, z in coordinates]
            for it in range(nt):
                for item in range(n_items):
                    itemdata = dfs.ReadItemTimeStep(item_numbers[item] + 1, it)
                    d = np.array([itemdata.Data[i] for i in indices])
                    d[d == deleteValue] = np.nan
                    data_list[item][:, it] = d

                t.append(startTime.AddSeconds(itemdata.Time).ToString("yyyy-MM-dd HH:mm:ss"))

        time = pd.DatetimeIndex(t)
        names = []
        for item in range(n_items):
            name = dfs.ItemInfo[item_numbers[item]].Name
            names.append(name)

        dfs.Close()

        return data_list, time, names


    def create_equidistant_calendar(self, dfs3file, data, start_time, timeseries_unit, dt, variable_type, unit, coordinate,
                                    x0, y0, length_x, length_y, names, title=None):
        """
        Creates a dfs3 file

        dfs3file:
            Location to write the dfs3 file
        data:
            list of matrices, one for each item. Matrix dimension: y, x, z, time
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
        y0:
            Lower right position
        length_x:
            length of each grid in the x direction (meters)
        length_y:
            length of each grid in the y direction (meters)
        names:
            array of names (ie. array of strings).
        title:
            title of the dfs2 file (can be blank)

        """

        if title is None:
            title = "dfs0 file"

        number_y = np.shape(data[0])[0]
        number_x = np.shape(data[0])[1]
        number_z = np.shape(data[0])[2]
        n_time_steps = np.shape(data[0])[3]
        n_items = len(data)

        if not all( np.shape(d)[0] == number_y for d in data):
            raise Warning("ERROR data matrices in the Y dimension do not all match in the data list. "
                     "Data is list of matices [y,x,time]")
        if not all(np.shape(d)[1] == number_x for d in data):
            raise Warning("ERROR data matrices in the X dimension do not all match in the data list. "
                     "Data is list of matices [y,x,time]")
        if not all(np.shape(d)[2] == number_z for d in data):
            raise Warning("ERROR data matrices in the X dimension do not all match in the data list. "
                     "Data is list of matices [y,x,time]")
        if not all(np.shape(d)[3] == n_time_steps for d in data):
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
        builder = Dfs3Builder.Create(title, 'Matlab DFS', 0);

        # Set up the header
        builder.SetDataType(1);
        builder.SetGeographicalProjection(factory.CreateProjectionGeoOrigin(coordinate[0], coordinate[1], coordinate[2], coordinate[3]))
        builder.SetTemporalAxis(
            factory.CreateTemporalEqCalendarAxis(timeseries_unit, system_start_time, 0, dt))
        builder.SetSpatialAxis(
            factory.CreateAxisEqD3(eumUnit.eumUmeter, number_x, x0, length_x, number_y, y0, length_y, number_z, 0, 1))

        deletevalue = builder.DeleteValueFloat

        for i in range(n_items):
            builder.AddDynamicItem(names[i], eumQuantity.Create(variable_type[i], unit[i]), DfsSimpleType.Float, DataValueType.Instantaneous)

        try:
            builder.CreateFile(dfs3file)
        except IOError:
            print('cannot create dfs3 file: ', dfs3file)

        dfs = builder.GetFile()

        for i in range(n_time_steps):
            for item in range(n_items):
                #d = data[item][:, :, :, i]
                #d.reshape(number_z, number_y, number_x).swapaxes(0, 2).swapaxes(0, 1)
                #d = np.flipud(d)
                #d[np.isnan(d)] = deletevalue
                #darray = Array[System.Single](np.array(d.reshape(d.size, 1)[:, 0]))
                #dfs.WriteItemTimeStepNext(0, darray)


                # TESTED AND WORKDS if data already in the y,x,z,t format
                d = data[item][:, :, :, i]
                d = d.swapaxes(0, 1)
                d = d.swapaxes(0, 2)
                d = np.fliplr(d)
                d[np.isnan(d)] = deletevalue
                darray = Array[System.Single](np.array(d.reshape(d.size, 1)[:, 0]))
                dfs.WriteItemTimeStepNext(0, darray)


        dfs.Close()

    def write(self, dfs3file, data):
        """
        Function: write to a pre-created dfs3 file. Only ONE item supported.

        NOTE:
            The dfs2 file must be pre-created with corresponding y,x, z dimensions and number of time steps.

        The Data Matrix
            size ( y, x, z, nt)

        usage:
            write( filename, data) where  data( y, x, z, nt)

        Returns:
            Nothing
        """

        # Open the dfs file for writing
        dfs = DfsFileFactory.Dfs3FileOpenEdit(dfs3file);

        # Determine the size of the grid
        yNum = dfs.SpatialAxis.YCount
        xNum = dfs.SpatialAxis.XCount
        zNum = dfs.SpatialAxis.ZCount

        nt = dfs.FileInfo.TimeAxis.NumberOfTimeSteps
        deletevalue = dfs.FileInfo.DeleteValueFloat

        if data.shape[0] is not yNum:
            sys.exit("ERROR Y dimension does not match")
        elif data.shape[1] is not xNum:
            sys.exit("ERROR X dimension does not match")
        elif data.shape[2] is not zNum:
            sys.exit("ERROR X dimension does not match")
        elif data.shape[3] is not nt:
            sys.exit("ERROR Number of Time Steps dimension does not match")

        for it in range(nt):
            d = data[:, :, :, it]
            d[np.isnan(d)] = deletevalue
            d = d.swapaxes(1, 2).swapaxes(1, 0)
            d = d[:,::-1,:]
            darray = Array[System.Single](np.asarray(d).reshape(-1))
            dfs.WriteItemTimeStepNext(0, darray)

        dfs.Close()
