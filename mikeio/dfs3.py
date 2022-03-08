import os
import numpy as np
from datetime import datetime, timedelta
from mikecore.eum import eumUnit, eumQuantity
from mikecore.DfsFileFactory import DfsFileFactory
from mikecore.DfsFactory import DfsFactory
from mikecore.DfsFile import DfsSimpleType, DataValueType
from mikecore.DfsBuilder import DfsBuilder

from .dfsutil import _valid_item_numbers, _valid_timesteps, _get_item_info
from .dataset import Dataset
from .eum import TimeStepUnit
from .dfs import _Dfs123


class Dfs3(_Dfs123):
    def __init__(self, filename=None, dtype=np.float64):
        super(Dfs3, self).__init__(filename, dtype)
        if filename:
            self._read_dfs3_header()

    def get_bottom_values(self):
        bottom2D = []
        data = self.read()
        bottom_data = np.nan * np.ones(shape=(self.shape[0],) + self.shape[2:])
        for item_number in range(self.n_items):
            b = np.nan * np.ones(self.shape[2:])
            for ts in range(self.n_timesteps):
                d = data.data[item_number][ts, ...]
                for layer in range(d.shape[0]):  # going from surface to bottom
                    y = d[layer, ...]
                    b[~np.isnan(y)] = y[~np.isnan(y)]
                bottom_data[ts, ...] = np.flipud(b)
            bottom2D.append(bottom_data)
        return bottom2D

    def _read_dfs3_header(self):
        if not os.path.isfile(self._filename):
            raise Exception(f"file {self._filename} does not exist!")

        self._dfs = DfsFileFactory.Dfs3FileOpen(self._filename)

        self.source = self._dfs
        self._dfs
        self._dx = self._dfs.SpatialAxis.Dx
        self._dy = self._dfs.SpatialAxis.Dy
        self._dz = self._dfs.SpatialAxis.Dz
        self._nx = self._dfs.SpatialAxis.XCount
        self._ny = self._dfs.SpatialAxis.YCount
        self._nz = self._dfs.SpatialAxis.ZCount
        self._read_header()

    def __calculate_index(self, nx, ny, nz, x, y, z):
        """Calculates the position in the dfs3 data array based on the
        number of x,y,z layers (nx,ny,nz) at the specified x,y,z position.

        Error checking is done here to see if the x,y,z coordinates are out of range.
        """
        if x >= nx:
            raise IndexError("x coordinate is off the grid: ", x)
        if y >= ny:
            raise IndexError("y coordinate is off the grid: ", y)
        if z >= nz:
            raise IndexError("z coordinate is off the grid: ", z)

        return y * nx + x + z * nx * ny

    def grid_coordinates(self, dfs3file):
        """Function: Returns the Grid information
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

        dfs = DfsFileFactory.DfsGenericOpen(dfs3file)

        # Determine the size of the grid
        axis = dfs.ItemInfo[0].SpatialAxis
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

    def read_slice(
        self,
        dfs3file,
        lower_left_xy,
        upper_right_xy,
        items=None,
        layers=None,
        conservative=True,
    ):
        """Function: Read data from a dfs3 file within the locations chosen


        Usage:
            [data,time,name] = read( filename, lower_left_xy, upper_right_xy, items, conservative)
        dfs3file
            a full path and filename to the dfs3 file
        lower_left_xy
            list or array of size two with the X and the Y coordinate (same projection as the dfs3)
        upper_right_xy
            list or array of size two with the X and the Y coordinate (same projection as the dfs3)
        items
            list of indices (base 0) to read from
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

        data = self.read(dfs3file, items=items, layers=layers)

        dfs = DfsFileFactory.DfsGenericOpen(dfs3file)

        # Determine the size of the grid
        axis = dfs.ItemInfo[0].SpatialAxis
        dx = axis.Dx
        dy = axis.Dy
        x0 = axis.X0
        y0 = axis.Y0
        yNum = axis.YCount
        xNum = axis.XCount

        top_left_y = y0 + (yNum + 1) * dy

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
            raise IndexError("lower_left_x_index < 0.")
            lower_left_x_index = 0

        if upper_right_y_index < 0:
            raise IndexError("upper_right_y_index < 0.")
            upper_right_y_index = 0

        if lower_left_y_index > yNum - 1:
            raise IndexError("lower_left_y_index > yNum - 1")
            lower_left_y_index = yNum - 1

        if upper_right_x_index > xNum - 1:
            raise IndexError("upper_right_x_index > xNum - 1")
            upper_right_x_index = xNum - 1

        for i in range(len(data[0])):
            data[0][i] = data[0][i][
                upper_right_y_index:lower_left_y_index,
                lower_left_x_index:upper_right_x_index,
                :,
                :,
            ]

        return data

    def read(
        self, items=None, layers=None, coordinates=None, time=None, time_steps=None
    ) -> Dataset:
        """Function: Read data from a dfs3 file

        Usage:
            [data,time,name] = read( filename, items, layers=None, coordinates=None)

        items
            list of indices (base 0) to read from. If None then all the items.
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
            Returns Dataset with data (t, z, y, x)

            1) If coordinates is selected, then only return data at those coordinates
            2) coordinates specified overules layers.
            3) layer counts from the bottom
        """

        # Open the dfs file for reading
        dfs = DfsFileFactory.DfsGenericOpen(self._filename)

        item_numbers = _valid_item_numbers(dfs.ItemInfo, items)
        n_items = len(item_numbers)

        if time_steps is not None:
            warnings.warn(
                FutureWarning(
                    "time_steps have been renamed to time, and will be removed in a future release"
                )
            )
            time = time_steps
        time_steps = _valid_timesteps(dfs.FileInfo, time)
        nt = len(time_steps)

        # Determine the size of the grid
        axis = dfs.ItemInfo[0].SpatialAxis
        zNum = axis.ZCount
        yNum = axis.YCount
        xNum = axis.XCount
        deleteValue = dfs.FileInfo.DeleteValueFloat

        data_list = []

        if coordinates is None:
            # if nt is 0, then the dfs is 'static' and must be handled differently
            if nt != 0:
                for item in range(n_items):
                    if layers is None:
                        # Initialize an empty data block
                        data = np.ndarray(shape=(nt, zNum, yNum, xNum), dtype=float)
                        data_list.append(data)
                    else:
                        data = np.ndarray(
                            shape=(nt, len(layers), yNum, xNum), dtype=float
                        )
                        data_list.append(data)

            else:
                raise ValueError(
                    "Static dfs3 files (with no time steps) are not supported."
                )

        else:
            ncoordinates = len(coordinates)
            for item in range(n_items):
                # Initialize an empty data block
                data = np.ndarray(shape=(nt, ncoordinates), dtype=float)
                data_list.append(data)

        t_seconds = np.zeros(nt, dtype=float)
        startTime = dfs.FileInfo.TimeAxis.StartDateTime

        if coordinates is None:
            for it_number, it in enumerate(time_steps):
                for item in range(n_items):
                    itemdata = dfs.ReadItemTimeStep(item_numbers[item] + 1, int(it))

                    src = itemdata.Data
                    # d = to_numpy(src)
                    d = src

                    # DO a direct copy instead of eleement by elment
                    d = d.reshape(zNum, yNum, xNum)  # .swapaxes(0, 2).swapaxes(0, 1)
                    d = np.flipud(d)  # TODO
                    d[d == deleteValue] = np.nan
                    if layers is None:
                        data_list[item][it_number, :, :, :] = d
                    else:
                        for l in range(len(layers)):
                            data_list[item][it_number, l, :, :] = d[layers[l], :, :]

                t_seconds[it_number] = itemdata.Time
        else:
            indices = [
                self.__calculate_index(xNum, yNum, zNum, x, y, z)
                for x, y, z in coordinates
            ]
            for it in range(nt):
                for item in range(n_items):
                    itemdata = dfs.ReadItemTimeStep(item_numbers[item] + 1, it)
                    d = np.array([itemdata.Data[i] for i in indices])
                    d[d == deleteValue] = np.nan
                    data_list[item][it, :] = d

                t_seconds[it] = itemdata.Time

        start_time = dfs.FileInfo.TimeAxis.StartDateTime
        time = [start_time + timedelta(seconds=tsec) for tsec in t_seconds]

        items = _get_item_info(dfs.ItemInfo, item_numbers)

        dfs.Close()

        return Dataset(data_list, time, items)

    def write(
        self,
        filename,
        data,
        start_time=None,
        dt=1,
        items=None,
        dx=1.0,
        dy=1.0,
        dz=1.0,
        x0=0,
        y0=0,
        coordinate=None,
        timeseries_unit=TimeStepUnit.SECOND,
        title=None,
    ):
        """
        Write a dfs3 file

        Parameters
        ----------

        filename: str
            Location to write the dfs3 file
        data: list[np.array]
            list of matrices, one for each item. Matrix dimension: time, z, y, x
        start_time: datetime, optional
            start date of type datetime.
        timeseries_unit: Timestep, optional
            TimeStep default TimeStep.SECOND
        dt: float, optional
            The time step. Therefore dt of 5.5 with timeseries_unit of TimeStep.MINUTE
            means 5 mins and 30 seconds. Default 1
        items: list[ItemInfo], optional
            List of ItemInfo corresponding to a variable types (ie. Water Level).
        coordinate:
            ['UTM-33', 12.4387, 55.2257, 327]  for UTM, Long, Lat, North to Y orientation. Note: long, lat in decimal degrees
        x0: float, optional
            Lower right position
        y0: float, optional
            Lower right position
        dx: float, optional
            length of each grid in the x direction (projection units)
        dy: float, optional
            length of each grid in the y direction (projection units)
        dz: float, optional
            length of each grid in the z direction (projection units)

        title: str, optional
            title of the dfs2 file. Default is blank.
        """

        if title is None:
            title = "dfs3 file"

        n_time_steps = np.shape(data[0])[0]
        number_z = np.shape(data[0])[1]
        number_y = np.shape(data[0])[2]
        number_x = np.shape(data[0])[3]

        n_items = len(data)

        system_start_time = start_time

        # Create an empty dfs3 file object
        factory = DfsFactory()
        builder = DfsBuilder(title, "mikeio", 0)

        # Set up the header
        builder.SetDataType(1)
        builder.SetGeographicalProjection(
            factory.CreateProjectionGeoOrigin(*coordinate)
        )
        builder.SetTemporalAxis(
            factory.CreateTemporalEqCalendarAxis(
                timeseries_unit, system_start_time, 0, dt
            )
        )
        builder.SetSpatialAxis(
            factory.CreateAxisEqD3(
                eumUnit.eumUmeter,
                number_x,
                x0,
                dx,
                number_y,
                y0,
                dy,
                number_z,
                0,
                dz,
            )
        )

        for i in range(n_items):
            builder.AddCreateDynamicItem(
                items[i].name,
                eumQuantity.Create(items[i].type, items[i].unit),
                DfsSimpleType.Float,
                DataValueType.Instantaneous,
            )

        try:
            builder.CreateFile(filename)
        except IOError:
            print("cannot create dfs3 file: ", filename)

        dfs = builder.GetFile()
        deletevalue = dfs.FileInfo.DeleteValueFloat  # -1.0000000031710769e-30

        for i in range(n_time_steps):
            for item in range(n_items):
                d = data[item][i]
                d[np.isnan(d)] = deletevalue
                d = np.flipud(d)  # TODO
                # darray = to_dotnet_float_array(d.reshape(d.size, 1)[:, 0])
                darray = d.reshape(d.size, 1)[:, 0].astype(np.float32)

                dfs.WriteItemTimeStepNext(0, darray)

        dfs.Close()

    @property
    def dx(self):
        """Step size in x direction"""
        return self._dx

    @property
    def dy(self):
        """Step size in y direction"""
        return self._dy

    @property
    def dz(self):
        """Step size in y direction"""
        return self._dz

    @property
    def shape(self):
        return (self._n_timesteps, self._nz, self._ny, self._nx)
