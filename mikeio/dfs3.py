import os
import warnings
import numpy as np
from mikecore.eum import eumUnit, eumQuantity
from mikecore.DfsFileFactory import DfsFileFactory
from mikecore.DfsFactory import DfsFactory
from mikecore.DfsFile import DfsSimpleType, DataValueType
from mikecore.DfsBuilder import DfsBuilder
import pandas as pd

from mikeio.spatial.geometry import GeometryUndefined

from .dfsutil import _valid_item_numbers, _valid_timesteps, _get_item_info
from .dataset import Dataset
from .eum import TimeStepUnit
from .dfs import _Dfs123
from .spatial.grid_geometry import Grid2D, Grid3D


class Dfs3(_Dfs123):

    _ndim = 3

    def __init__(self, filename=None, dtype=np.float32):
        super(Dfs3, self).__init__(filename, dtype)

        self._dx = None
        self._dy = None
        self._dz = None
        self._nx = None
        self._ny = None
        self._nz = None
        self._x0 = 0
        self._y0 = 0
        self._z0 = 0
        self.geometry = None

        if filename:
            self._read_dfs3_header()
            self.geometry = Grid3D(
                x0=self._x0,
                dx=self._dx,
                nx=self._nx,
                y0=self._y0,
                dy=self._dy,
                ny=self._ny,
                z0=self._z0,
                dz=self._dz,
                nz=self._nz,
                origin=(self._longitude, self._latitude),
                projection=self._projstr,
            )

    def __repr__(self):
        out = ["<mikeio.Dfs3>"]

        if os.path.isfile(self._filename):
            out.append(f"geometry: {self.geometry}")

            if self._n_items is not None:
                if self._n_items < 10:
                    out.append("items:")
                    for i, item in enumerate(self.items):
                        out.append(f"  {i}:  {item}")
                else:
                    out.append(f"number of items: {self._n_items}")

                if self._n_timesteps == 1:
                    out.append("time: time-invariant file (1 step)")
                else:
                    out.append(f"time: {self._n_timesteps} steps")
                    out.append(f"start time: {self._start_time}")

        return str.join("\n", out)

    def _read_dfs3_header(self):
        if not os.path.isfile(self._filename):
            raise Exception(f"file {self._filename} does not exist!")

        self._dfs = DfsFileFactory.Dfs3FileOpen(self._filename)

        self.source = self._dfs

        self._x0 = self._dfs.SpatialAxis.X0
        self._y0 = self._dfs.SpatialAxis.Y0
        self._z0 = self._dfs.SpatialAxis.Z0
        self._dx = self._dfs.SpatialAxis.Dx
        self._dy = self._dfs.SpatialAxis.Dy
        self._dz = self._dfs.SpatialAxis.Dz
        self._nx = self._dfs.SpatialAxis.XCount
        self._ny = self._dfs.SpatialAxis.YCount
        self._nz = self._dfs.SpatialAxis.ZCount
        self._read_header()

    def read(
        self,
        *,
        items=None,
        time=None,
        time_steps=None,
        area=None,
        layers=None,
    ) -> Dataset:

        if area is not None:
            return NotImplementedError("area subsetting is not yet implemented")

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
        zNum = self.geometry.nz
        yNum = self.geometry.ny
        xNum = self.geometry.nx
        deleteValue = dfs.FileInfo.DeleteValueFloat

        data_list = []

        layers = None if layers is None else np.atleast_1d(layers)
        geometry = self._geometry_for_layers(layers, self.geometry)

        nz = zNum if layers is None else len(layers)
        shape = (nt, nz, yNum, xNum) if nz > 1 else (nt, yNum, xNum)
        for item in range(n_items):
            data = np.ndarray(shape=shape, dtype=float)
            data_list.append(data)

        t_seconds = np.zeros(nt, dtype=float)

        for it_number, it in enumerate(time_steps):
            for item in range(n_items):
                itemdata = dfs.ReadItemTimeStep(item_numbers[item] + 1, int(it))
                d = itemdata.Data

                d = d.reshape(zNum, yNum, xNum)
                d[d == deleteValue] = np.nan

                if layers is None:
                    data_list[item][it_number, :, :, :] = d
                elif len(layers) == 1:
                    data_list[item][it_number, :, :] = d[layers[0], :, :]
                else:
                    for l in range(len(layers)):
                        data_list[item][it_number, l, :, :] = d[layers[l], :, :]

            t_seconds[it_number] = itemdata.Time

        dfs.Close()

        time = pd.to_datetime(t_seconds, unit="s", origin=self.start_time)
        items = _get_item_info(dfs.ItemInfo, item_numbers)
        return Dataset(data_list, time=time, items=items, geometry=geometry)

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
        dz=None,
        coordinate=None,
        title=None,
    ):
        """
        Create a dfs3 file

        Parameters
        ----------

        filename: str
            Location to write the dfs3 file
        data: list[np.array] or Dataset
            list of matrices, one for each item. Matrix dimension: time, y, x
        start_time: datetime, optional, deprecated
            start date of type datetime.
        dt: float, optional
            The time step in seconds.
        datetimes: datetime, optional, deprecated
            The list of datetimes for the case of non-equisstant Timeaxis.
        items: list[ItemInfo], optional
            List of ItemInfo corresponding to a variable types (ie. Water Level).
        dx: float, optional
            length of each grid in the x direction (projection units)
        dy: float, optional
            length of each grid in the y direction (projection units)
        dz: float, optional
            length of each grid in the z direction (projection units)
        coordinate:
            list of [projection, origin_x, origin_y, orientation]
            e.g. ['LONG/LAT', 12.4387, 55.2257, 327]
        title: str, optional
            title of the dfs3 file. Default is blank.
        keep_open: bool, optional
            Keep file open for appending
        """

        if start_time:
            warnings.warn(
                "setting start_time is deprecated, please supply data in the form of a Dataset",
                FutureWarning,
            )

        if datetimes:
            warnings.warn(
                "setting datetimes is deprecated, please supply data in the form of a Dataset",
                FutureWarning,
            )

        if items:
            warnings.warn(
                "setting items is deprecated, please supply data in the form of a Dataset",
                FutureWarning,
            )

        if isinstance(data, list):
            warnings.warn(
                "supplying data as a list of numpy arrays is deprecated, please supply data in the form of a Dataset",
                FutureWarning,
            )

        filename = str(filename)

        self._builder = DfsBuilder.Create(title, "mikeio", 0)
        if not self._dx:
            self._dx = 1
        if dx:
            self._dx = dx

        if not self._dy:
            self._dy = 1
        if dy:
            self._dy = dy

        if not self._dz:
            self._dz = 1
        if dz:
            self._dz = dz

        self._write(
            filename,
            data,
            start_time,
            dt,
            datetimes,
            items,
            coordinate,
            title,
        )

    def _set_spatial_axis(self):
        self._builder.SetSpatialAxis(
            self._factory.CreateAxisEqD3(
                eumUnit.eumUmeter,
                self._nx,
                self._x0,
                self._dx,
                self._ny,
                self._y0,
                self._dy,
                self._nz,
                self._z0,
                self._dz,
            )
        )

    @staticmethod
    def _geometry_for_layers(layers, geometry):
        if layers is not None:
            g = geometry
            if len(layers) == 1:
                geometry = Grid2D(
                    x=g.x + g._origin[0],
                    y=g.y + g._origin[1],
                    projection=g.projection,
                )
            else:
                d = np.diff(g.z[layers])
                if np.any(d < 1) or not np.allclose(d, d[0]):
                    warnings.warn(
                        "Extracting non-equidistant layers! Cannot use Grid3D."
                    )
                    geometry = GeometryUndefined()
                else:
                    geometry = Grid3D(
                        x=g.x,
                        y=g.y,
                        z=g.z[layers],
                        origin=g._origin,
                        projection=g.projection,
                    )
        return geometry

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
