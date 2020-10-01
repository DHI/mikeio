import os
import numpy as np
from datetime import timedelta
from DHI.Generic.MikeZero import eumUnit
from DHI.Generic.MikeZero.DFS import (
    DfsFileFactory,
    DfsFactory,
    DfsSimpleType,
    DataValueType,
)
from DHI.Generic.MikeZero.DFS.dfs123 import Dfs2Builder
from DHI.Projections import Cartography

from .dutil import Dataset, get_item_info, get_valid_items_and_timesteps
from .dotnet import (
    to_numpy,
    to_dotnet_float_array,
    to_dotnet_datetime,
    from_dotnet_datetime,
)
from .eum import ItemInfo
from .helpers import safe_length
from .dfs import Dfs123


class Dfs2(Dfs123):

    _dx = None
    _dy = None

    def __init__(self, filename=None):
        super(Dfs2, self).__init__(filename)

        if filename:
            self._read_dfs2_header()

    def __repr__(self):
        out = ["Dfs2"]

        if self._filename:
            out.append(f"dx: {self.dx:.5f}")
            out.append(f"dy: {self.dy:.5f}")

        if self._n_items is not None:
            if self._n_items < 10:
                out.append("Items:")
                for i, item in enumerate(self.items):
                    out.append(f"  {i}:  {item}")
            else:
                out.append(f"Number of items: {self._n_items}")
        if self._filename:
            if self._n_timesteps == 1:
                out.append(f"Time: time-invariant file (1 step)")
            else:
                out.append(f"Time: {self._n_timesteps} steps")
                out.append(f"Start time: {self._start_time}")

        return str.join("\n", out)

    def _read_dfs2_header(self):
        if not os.path.isfile(self._filename):
            raise Exception(f"file {self._filename} does not exist!")

        dfs = DfsFileFactory.Dfs2FileOpen(self._filename)
        self._dx = dfs.SpatialAxis.Dx
        self._dy = dfs.SpatialAxis.Dy

        self._read_header(dfs)

    def find_nearest_element(
        self, lon, lat,
    ):
        """Find index of closest element

        Parameters
        ----------

        lon: float
            longitude
        lat: float
            latitude

        Returns
        -------

        (int,int): indexes in y, x 
        """
        projection = self._dfs.FileInfo.Projection
        axis = self._dfs.SpatialAxis
        cart = Cartography(
            projection.WKTString,
            projection.Longitude,
            projection.Latitude,
            projection.Orientation,
        )

        # C# out parameters must be handled in special way
        (_, xx, yy) = cart.Geo2Xy(lon, lat, 0.0, 0.0)

        j = int(xx / axis.Dx + 0.5)
        k = axis.YCount - int(yy / axis.Dy + 0.5) - 1

        j = min(max(0, j), axis.XCount - 1)
        k = min(max(0, k), axis.YCount - 1)

        return k, j

    def read(self, items=None, time_steps=None):
        """
        Read data from a dfs2 file
        
        Parameters
        ---------
        items: list[int] or list[str], optional
            Read only selected items, by number (0-based), or by name
        time_steps: int or list[int], optional
            Read only selected time_steps

        Returns
        -------
        Dataset
            A dataset with data dimensions [t,y,x]
        """
        dfs = DfsFileFactory.Dfs2FileOpen(self._filename)
        self._dfs = dfs
        self._source = dfs

        items, item_numbers, time_steps = get_valid_items_and_timesteps(
            self, items, time_steps
        )

        # Determine the size of the grid
        axis = dfs.SpatialAxis
        ny = axis.YCount
        nx = axis.XCount

        for t in time_steps:
            if t > (self.n_timesteps - 1):
                raise ValueError(
                    f"Trying to read timestep {t}: max timestep is {self.n_timesteps-1}"
                )

        n_items = len(item_numbers)
        nt = len(time_steps)

        data_list = [
            np.ndarray(shape=(nt, ny, nx), dtype=float) for item in range(n_items)
        ]

        t_seconds = np.zeros(len(time_steps), dtype=float)

        for i in range(nt):
            it = time_steps[i]
            for item in range(n_items):

                itemdata = dfs.ReadItemTimeStep(item_numbers[item] + 1, it)

                src = itemdata.Data
                d = to_numpy(src)

                d = d.reshape(ny, nx)
                d = np.flipud(d)
                d[d == self.deletevalue] = np.nan
                data_list[item][i, :, :] = d

            t_seconds[i] = itemdata.Time

        time = [self.start_time + timedelta(seconds=t) for t in t_seconds]

        items = get_item_info(dfs, item_numbers)

        dfs.Close()
        return Dataset(data_list, time, items)

    def write(
        self,
        filename,
        data,
        start_time=None,
        dt=None,
        datetimes=None,
        items=None,
        dx=None,
        dy=None,
        x0=0,
        y0=0,
        coordinate=None,
        title=None,
    ):
        """
        Create a dfs2 file

        Parameters
        ----------

        filename: str
            Location to write the dfs2 file
        data: list[np.array] or Dataset
            list of matrices, one for each item. Matrix dimension: time, y, x
        start_time: datetime, optional
            start date of type datetime.
        dt: float, optional
            The time step in seconds.
        datetimes: list[datetime], optional
            datetimes, creates a non-equidistant calendar axis
        items: list[ItemInfo], optional
            List of ItemInfo corresponding to a variable types (ie. Water Level).
        x0: float, optional
            Lower right position
        x0: float, optional
            Lower right position
        dx: float, optional
            length of each grid in the x direction (projection units)
        dy: float, optional
            length of each grid in the y direction (projection units)
        coordinate:
            ['UTM-33', 12.4387, 55.2257, 327]  for UTM, Long, Lat, North to Y orientation. Note: long, lat in decimal degrees
        title: str, optional
            title of the dfs2 file. Default is blank.
        """

        self._write_handle_common_arguments(
            title, data, items, coordinate, start_time, dt
        )

        number_y = np.shape(data[0])[1]
        number_x = np.shape(data[0])[2]

        if dx is None:
            if self._dx is not None:
                dx = self._dx
            else:
                dx = 1

        if dy is None:
            if self._dy is not None:
                dy = self._dy
            else:
                dy = 1

        if not all(np.shape(d)[0] == self._n_time_steps for d in data):
            raise ValueError(
                "ERROR data matrices in the time dimension do not all match in the data list. "
                "Data is list of matrices [t,y,x]"
            )
        if not all(np.shape(d)[1] == number_y for d in data):
            raise ValueError(
                "ERROR data matrices in the Y dimension do not all match in the data list. "
                "Data is list of matrices [t,y,x]"
            )
        if not all(np.shape(d)[2] == number_x for d in data):
            raise ValueError(
                "ERROR data matrices in the X dimension do not all match in the data list. "
                "Data is list of matrices [t,y,x]"
            )

        if datetimes is None:
            self._is_equidistant = True
        else:
            self._is_equidistant = False
            start_time = datetimes[0]
            self._start_time = start_time

        factory = DfsFactory()
        builder = Dfs2Builder.Create(title, "mikeio", 0)

        self._builder = builder
        self._factory = factory

        builder.SetSpatialAxis(
            factory.CreateAxisEqD2(
                eumUnit.eumUmeter, number_x, x0, dx, number_y, y0, dy
            )
        )

        dfs = self._setup_header(filename)
        # coordinate, start_time, dt, timeseries_unit, items, filename
        # )

        deletevalue = dfs.FileInfo.DeleteValueFloat  # -1.0000000031710769e-30

        for i in range(self._n_time_steps):
            for item in range(self._n_items):
                d = self._data[item][i, :, :]
                d[np.isnan(d)] = deletevalue
                d = d.reshape(number_y, number_x)
                d = np.flipud(d)
                darray = to_dotnet_float_array(d.reshape(d.size, 1)[:, 0])

                if self._is_equidistant:
                    dfs.WriteItemTimeStepNext(0, darray)
                else:
                    t = datetimes[i]
                    relt = (t - self._start_time).total_seconds()
                    dfs.WriteItemTimeStepNext(relt, darray)

        dfs.Close()

    @property
    def dx(self):
        """Step size in x direction
        """
        return self._dx

    @property
    def dy(self):
        """Step size in y direction
        """
        return self._dy

