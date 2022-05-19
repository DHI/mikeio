import os
import warnings
import numpy as np
from mikecore.eum import eumUnit, eumQuantity
from mikecore.DfsFileFactory import DfsFileFactory
from mikecore.DfsFactory import DfsFactory
from mikecore.DfsFile import DfsSimpleType, DataValueType, DfsFile
from mikecore.DfsBuilder import DfsBuilder
from mikecore.Projections import Cartography
import pandas as pd

from . import __dfs_version__
from .dfsutil import _valid_item_numbers, _valid_timesteps, _get_item_info
from .dataset import Dataset
from .eum import TimeStepUnit
from .dfs import _Dfs123
from .spatial.grid_geometry import Grid2D, Grid3D
from .spatial.geometry import GeometryUndefined


def write_dfs3(filename: str, ds: Dataset, title="") -> None:
    dfs = _write_dfs3_header(filename, ds, title)
    _write_dfs3_data(dfs, ds)


def _write_dfs3_header(filename, ds: Dataset, title="") -> DfsFile:
    builder = DfsBuilder.Create(title, "mikeio", __dfs_version__)
    builder.SetDataType(0)

    geometry: Grid3D = ds.geometry

    factory = DfsFactory()
    _write_dfs3_spatial_axis(builder, factory, geometry)
    origin = geometry._origin  # Origin in geographical coordinates
    orient = geometry._orientation

    # if geometry.is_geo:
    proj = factory.CreateProjectionGeoOrigin(
        geometry.projection_string, *origin, orient
    )
    # else:
    #    cart: Cartography = Cartography.CreateProjOrigin(
    #        geometry.projection_string, *origin, orient
    #    )
    #    proj = factory.CreateProjectionGeoOrigin(
    #        wktProjectionString=geometry.projection,
    #        lon0=cart.LonOrigin,
    #        lat0=cart.LatOrigin,
    #        orientation=cart.Orientation,
    #    )

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


def _write_dfs3_data(dfs: DfsFile, ds: Dataset) -> None:

    deletevalue = dfs.FileInfo.DeleteValueFloat  # ds.deletevalue
    t_rel = 0
    for i in range(ds.n_timesteps):
        for item in range(ds.n_items):

            if "time" not in ds.dims:
                d = ds[item].values
            else:
                d = ds[item].values[i]
            d = d.copy()  # to avoid modifying the input
            d[np.isnan(d)] = deletevalue

            d = d.reshape(ds.shape[-3:])  # spatial axes
            darray = d.flatten()

            if not ds.is_equidistant:
                t_rel = (ds.time[i] - ds.time[0]).total_seconds()

            dfs.WriteItemTimeStepNext(t_rel, darray.astype(np.float32))

    dfs.Close()


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
                orientation=self._orientation,
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

        if layers == "top":
            layers = -1
        # if layers == "bottom":
        #    return NotImplementedError()
        layers = None if layers is None else np.atleast_1d(layers)
        geometry = self.geometry._geometry_for_layers(layers)

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
                    if layers[0] == "bottom":
                        data_list[item][it_number, :, :] = self._get_bottom_values(d)
                    else:
                        data_list[item][it_number, :, :] = d[layers[0], :, :]
                else:
                    for l in range(len(layers)):
                        data_list[item][it_number, l, :, :] = d[layers[l], :, :]

            t_seconds[it_number] = itemdata.Time

        dfs.Close()

        time = pd.to_datetime(t_seconds, unit="s", origin=self.start_time)
        items = _get_item_info(dfs.ItemInfo, item_numbers)
        return Dataset(
            data_list, time=time, items=items, geometry=geometry, validate=False
        )

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
