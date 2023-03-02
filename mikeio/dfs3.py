import os

import numpy as np
import pandas as pd

from mikecore.DfsBuilder import DfsBuilder
from mikecore.DfsFactory import DfsFactory
from mikecore.DfsFile import DfsFile, DfsSimpleType
from mikecore.DfsFileFactory import DfsFileFactory
from mikecore.eum import eumQuantity, eumUnit
from mikecore.Projections import Cartography

from . import __dfs_version__
from .dataset import Dataset
from .dfs import (
    _Dfs123,
    _get_item_info,
    _valid_item_numbers,
    _valid_timesteps,
    _write_dfs_data,
)
from .eum import TimeStepUnit
from .spatial.grid_geometry import Grid3D


def write_dfs3(filename: str, ds: Dataset, title="") -> None:
    dfs = _write_dfs3_header(filename, ds, title)
    _write_dfs_data(dfs=dfs, ds=ds, n_spatial_dims=3)


def _write_dfs3_header(filename, ds: Dataset, title="") -> DfsFile:
    builder = DfsBuilder.Create(title, "mikeio", __dfs_version__)
    builder.SetDataType(0)

    geometry: Grid3D = ds.geometry

    factory = DfsFactory()
    _write_dfs3_spatial_axis(builder, factory, geometry)
    origin = geometry.origin  # Origin in geographical coordinates
    orient = geometry.orientation

    if geometry.is_geo:
        proj = factory.CreateProjectionGeoOrigin(
            geometry.projection_string, *origin, orient
        )
    else:
        cart: Cartography = Cartography.CreateProjOrigin(
            geometry.projection_string, *origin, orient
        )
        proj = factory.CreateProjectionGeoOrigin(
            wktProjectionString=geometry.projection,
            lon0=cart.LonOrigin,
            lat0=cart.LatOrigin,
            orientation=cart.Orientation,
        )

    builder.SetGeographicalProjection(proj)

    timestep_unit = TimeStepUnit.SECOND
    dt = ds.timestep or 1.0  # It can not be None
    if ds.is_equidistant:
        time_axis = factory.CreateTemporalEqCalendarAxis(
            timestep_unit, ds.time[0], 0, dt
        )
    else:
        time_axis = factory.CreateTemporalNonEqCalendarAxis(timestep_unit, ds.time[0])
    builder.SetTemporalAxis(time_axis)

    for item in ds.items:
        builder.AddCreateDynamicItem(
            item.name,
            eumQuantity.Create(item.type, item.unit),
            DfsSimpleType.Float,
            item.data_value_type,
        )

    try:
        builder.CreateFile(filename)
    except IOError:
        print("cannot create dfs file: ", filename)

    return builder.GetFile()


def _write_dfs3_spatial_axis(builder, factory, geometry: Grid3D):
    builder.SetSpatialAxis(
        factory.CreateAxisEqD3(
            eumUnit.eumUmeter,
            geometry.nx,
            geometry.x[0],
            geometry.dx,
            geometry.ny,
            geometry.y[0],
            geometry.dy,
            geometry.nz,
            geometry.z[0],
            geometry.dz,
        )
    )


class Dfs3(_Dfs123):

    _ndim = 3

    def __init__(self, filename=None):
        super().__init__(filename)

        self._dx = None
        self._dy = None
        self._dz = None
        self._nx = None
        self._ny = None
        self._nz = None
        self._x0 = 0.0
        self._y0 = 0.0
        self._z0 = 0.0
        self.geometry = None

        if filename:
            self._read_dfs3_header()
            self._validate_no_orientation_in_geo()
            origin, orientation = self._origin_and_orientation_in_CRS()

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
                origin=origin,
                projection=self._projstr,
                orientation=orientation,
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

    def _read_dfs3_header(self, read_x0y0z0: bool = False):
        if not os.path.isfile(self._filename):
            raise Exception(f"file {self._filename} does not exist!")

        self._dfs = DfsFileFactory.Dfs3FileOpen(self._filename)

        self._source = self._dfs

        if read_x0y0z0:
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
        area=None,
        layers=None,
        keepdims=False,
    ) -> Dataset:
        """
        Read data from a dfs3 file

        Parameters
        ---------
        items: list[int] or list[str], optional
            Read only selected items, by number (0-based), or by name
        time: int, str, datetime, pd.TimeStamp, sequence, slice or pd.DatetimeIndex, optional
            Read only selected time steps, by default None (=all)
        keepdims: bool, optional
            When reading a single time step or a single layer only,
            should the singleton dimension be kept
            in the returned Dataset? by default: False
        layers: int, str, list[int], optional
            Read only data for specific layers, by default None

        Returns
        -------
        Dataset
        """

        if area is not None:
            return NotImplementedError(
                "area subsetting is not yet implemented for Dfs3"
            )
        # NOTE:
        # if keepdims is not False:
        #    return NotImplementedError("keepdims is not yet implemented for Dfs3")

        # Open the dfs file for reading
        dfs = DfsFileFactory.DfsGenericOpen(self._filename)

        item_numbers = _valid_item_numbers(dfs.ItemInfo, items)
        n_items = len(item_numbers)

        single_time_selected, time_steps = _valid_timesteps(dfs.FileInfo, time)
        nt = len(time_steps) if not single_time_selected else 1

        # Determine the size of the grid
        zNum = self.geometry.nz
        yNum = self.geometry.ny
        xNum = self.geometry.nx
        deleteValue = dfs.FileInfo.DeleteValueFloat

        data_list = []

        if layers == "top":
            layers = -1
        layers = None if layers is None else np.atleast_1d(layers)

        nz = zNum if layers is None else len(layers)
        if nz == 1 and (not keepdims):
            geometry = self.geometry._geometry_for_layers([0])
            dims = ("time", "y", "x")
            shape = (nt, yNum, xNum)
        else:
            geometry = self.geometry._geometry_for_layers(layers, keepdims)
            dims = ("time", "z", "y", "x")
            shape = (nt, nz, yNum, xNum)

        for item in range(n_items):
            data = np.ndarray(shape=shape, dtype=float)
            data_list.append(data)

        if single_time_selected and not keepdims:
            shape = shape[1:]
            dims = tuple([d for d in dims if d != "time"])

        t_seconds = np.zeros(nt, dtype=float)

        for i, it in enumerate(time_steps):
            for item in range(n_items):
                itemdata = dfs.ReadItemTimeStep(item_numbers[item] + 1, int(it))
                d = itemdata.Data

                d = d.reshape(zNum, yNum, xNum)
                d[d == deleteValue] = np.nan

                if layers is None:
                    dd = d
                elif len(layers) == 1:
                    if layers[0] == "bottom":
                        dd = self._get_bottom_values(d)
                    else:
                        dd = d[layers[0], :, :]
                else:
                    dd = d[layers, :, :]

                if single_time_selected and not keepdims:
                    data_list[item] = dd
                else:
                    data_list[item][i, ...] = dd

            t_seconds[i] = itemdata.Time

        dfs.Close()

        time = pd.to_datetime(t_seconds, unit="s", origin=self.start_time)
        items = _get_item_info(dfs.ItemInfo, item_numbers)
        return Dataset(
            data_list,
            time=time,
            items=items,
            geometry=geometry,
            dims=dims,
            validate=False,
        )

    def write(
        self,
        filename,
        data,
        dt=None,
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
        data: Dataset
            list of matrices, one for each item. Matrix dimension: time, y, x
        dt: float, optional
            The time step in seconds.
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

        if isinstance(data, list):
            raise TypeError(
                "supplying data as a list of numpy arrays is deprecated, please supply data in the form of a Dataset"
            )

        filename = str(filename)

        self._builder = DfsBuilder.Create(title, "mikeio", __dfs_version__)
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
            filename=filename,
            data=data,
            dt=dt,
            coordinate=coordinate,
            title=title,
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
    def _get_bottom_values(data):

        assert len(data.shape) == 3
        b = np.empty_like(data[0])
        b[:] = np.nan
        data = np.flipud(data)
        for layer in range(data.shape[0]):  # going from surface to bottom
            y = data[layer, ...]
            b[~np.isnan(y)] = y[~np.isnan(y)]

        return b

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

    @property
    def is_geo(self):
        """Are coordinates geographical (LONG/LAT)?"""
        return self._projstr == "LONG/LAT"
