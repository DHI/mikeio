import os
from copy import deepcopy
import numpy as np
import pandas as pd
from tqdm import tqdm

from mikecore.DfsFactory import DfsBuilder, DfsFactory
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
from .spatial.grid_geometry import Grid2D


def write_dfs2(filename: str, ds: Dataset, title="") -> None:
    dfs = _write_dfs2_header(filename, ds, title)
    _write_dfs_data(dfs=dfs, ds=ds, n_spatial_dims=2)


def _write_dfs2_header(filename, ds: Dataset, title="") -> DfsFile:
    builder = DfsBuilder.Create(title, "mikeio", __dfs_version__)
    builder.SetDataType(0)

    geometry: Grid2D = ds.geometry

    if (
        geometry._shift_origin_on_write
        and not geometry._is_rotated
        and not geometry.is_spectral
    ):
        geometry = deepcopy(ds.geometry)
        geometry._shift_x0y0_to_origin()

    factory = DfsFactory()
    _write_dfs2_spatial_axis(builder, factory, geometry)
    proj_str = geometry.projection_string
    origin = geometry.origin
    orient = geometry.orientation

    if geometry.is_geo:
        proj = factory.CreateProjectionGeoOrigin(proj_str, *origin, orient)
    else:
        cart: Cartography = Cartography.CreateProjOrigin(proj_str, *origin, orient)
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


def _write_dfs2_spatial_axis(builder, factory, geometry):
    builder.SetSpatialAxis(
        factory.CreateAxisEqD2(
            eumUnit.eumUmeter,
            geometry._nx,
            geometry._x0,
            geometry._dx,
            geometry._ny,
            geometry._y0,
            geometry._dy,
        )
    )


class Dfs2(_Dfs123):

    _ndim = 2

    def __init__(self, filename=None, type: str = "horizontal"):
        super().__init__(filename)

        self._dx = None
        self._dy = None
        self._nx = None
        self._ny = None
        self._x0 = 0.0
        self._y0 = 0.0
        self.geometry = None

        if filename:
            is_spectral = type.lower() in ["spectral", "spectra", "spectrum"]
            self._read_dfs2_header(read_x0y0=is_spectral)
            self._validate_no_orientation_in_geo()
            origin, orientation = self._origin_and_orientation_in_CRS()

            self.geometry = Grid2D(
                dx=self._dx,
                dy=self._dy,
                nx=self._nx,
                ny=self._ny,
                x0=self._x0,
                y0=self._y0,
                orientation=orientation,
                origin=origin,
                projection=self._projstr,
                is_spectral=is_spectral,
            )

    def __repr__(self):
        out = ["<mikeio.Dfs2>"]

        if os.path.isfile(self._filename):
            out.append(f"dx: {self.dx:.5f}")
            out.append(f"dy: {self.dy:.5f}")

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

    def _read_dfs2_header(self, read_x0y0: bool = False):
        if not os.path.isfile(self._filename):
            raise Exception(f"file {self._filename} does not exist!")

        self._dfs = DfsFileFactory.Dfs2FileOpen(self._filename)
        self._source = self._dfs
        if read_x0y0:
            self._x0 = self._dfs.SpatialAxis.X0
            self._y0 = self._dfs.SpatialAxis.Y0
        self._dx = self._dfs.SpatialAxis.Dx
        self._dy = self._dfs.SpatialAxis.Dy
        self._nx = self._dfs.SpatialAxis.XCount
        self._ny = self._dfs.SpatialAxis.YCount
        if self._dfs.FileInfo.TimeAxis.TimeAxisType == 4:
            self._is_equidistant = False

        self._read_header()

    def read(
        self,
        *,
        items=None,
        time=None,
        area=None,
        keepdims=False,
        dtype=np.float32,
    ) -> Dataset:
        """
        Read data from a dfs2 file

        Parameters
        ---------
        items: list[int] or list[str], optional
            Read only selected items, by number (0-based), or by name
        time: int, str, datetime, pd.TimeStamp, sequence, slice or pd.DatetimeIndex, optional
            Read only selected time steps, by default None (=all)
        keepdims: bool, optional
            When reading a single time step only, should the time-dimension be kept
            in the returned Dataset? by default: False
        area: array[float], optional
            Read only data inside (horizontal) area given as a
            bounding box (tuple with left, lower, right, upper) coordinates

        Returns
        -------
        Dataset
        """

        self._open()

        item_numbers = _valid_item_numbers(self._dfs.ItemInfo, items)
        n_items = len(item_numbers)
        items = _get_item_info(self._dfs.ItemInfo, item_numbers)

        single_time_selected, time_steps = _valid_timesteps(self._dfs.FileInfo, time)
        nt = len(time_steps) if not single_time_selected else 1

        if area is not None:
            take_subset = True
            ii, jj = self.geometry.find_index(area=area)
            shape = (nt, len(jj), len(ii))
            geometry = self.geometry._index_to_Grid2D(ii, jj)
        else:
            take_subset = False
            shape = (nt, self._ny, self._nx)
            geometry = self.geometry

        if single_time_selected and not keepdims:
            shape = shape[1:]

        data_list = [np.ndarray(shape=shape, dtype=dtype) for item in range(n_items)]

        t_seconds = np.zeros(len(time_steps))

        for i, it in enumerate(tqdm(time_steps, disable=not self.show_progress)):
            for item in range(n_items):

                itemdata = self._dfs.ReadItemTimeStep(item_numbers[item] + 1, int(it))
                d = itemdata.Data

                d[d == self.deletevalue] = np.nan
                d = d.reshape(self._ny, self._nx)

                if take_subset:
                    d = np.take(np.take(d, jj, axis=0), ii, axis=-1)

                if single_time_selected and not keepdims:
                    data_list[item] = d
                else:
                    data_list[item][i] = d

            t_seconds[i] = itemdata.Time

        self._dfs.Close()

        time = pd.to_datetime(t_seconds, unit="s", origin=self.start_time)

        if single_time_selected and not keepdims:
            dims = ("y", "x")
        else:
            dims = ("time", "y", "x")

        return Dataset(
            data_list,
            time=time,
            items=items,
            geometry=geometry,
            dims=dims,
            validate=False,
        )

    def _open(self):
        self._dfs = DfsFileFactory.Dfs2FileOpen(self._filename)
        self._source = self._dfs

    def write(
        self,
        filename,
        data,
        dt=None,
        dx=None,
        dy=None,
        title=None,
        keep_open=False,
    ):
        """
        Create a dfs2 file

        Parameters
        ----------

        filename: str
            Location to write the dfs2 file
        data: Dataset
            list of matrices, one for each item. Matrix dimension: time, y, x
        dt: float, optional
            The time step in seconds.
        dx: float, optional
            length of each grid in the x direction (projection units)
        dy: float, optional
            length of each grid in the y direction (projection units)
        title: str, optional
            title of the dfs2 file. Default is blank.
        keep_open: bool, optional
            Keep file open for appending
        """
        if isinstance(data, list):
            raise TypeError(
                "supplying data as a list of numpy arrays is deprecated, please supply data in the form of a Dataset",
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

        self._write(
            filename=filename,
            data=data,
            dt=dt,
            title=title,
            keep_open=keep_open,
        )

        if keep_open:
            return self

    def _set_spatial_axis(self):
        self._builder.SetSpatialAxis(
            self._factory.CreateAxisEqD2(
                eumUnit.eumUmeter,
                self._nx,
                self._x0,
                self._dx,
                self._ny,
                self._y0,
                self._dy,
            )
        )

    @property
    def x0(self):
        """Start point of x values (often 0)"""
        return self._x0

    @property
    def y0(self):
        """Start point of y values (often 0)"""
        return self._y0

    @property
    def dx(self):
        """Step size in x direction"""
        return self._dx

    @property
    def dy(self):
        """Step size in y direction"""
        return self._dy

    @property
    def shape(self):
        """Tuple with number of values in the t-, y-, x-direction"""
        return (self._n_timesteps, self._ny, self._nx)

    @property
    def nx(self):
        """Number of values in the x-direction"""
        return self._nx

    @property
    def ny(self):
        """Number of values in the y-direction"""
        return self._ny

    @property
    def is_geo(self):
        """Are coordinates geographical (LONG/LAT)?"""
        return self._projstr == "LONG/LAT"
