from copy import deepcopy
import os
import numpy as np
import warnings
from mikecore.eum import eumUnit, eumQuantity
from mikecore.DfsFile import DfsSimpleType, DfsFile
from mikecore.DfsFileFactory import DfsFileFactory
from mikecore.DfsFactory import DfsBuilder, DfsFactory
from mikecore.Projections import Cartography
import pandas as pd
from tqdm import tqdm

from . import __dfs_version__
from .dfs import _Dfs123
from .dataset import Dataset
from .eum import TimeStepUnit
from .spatial.grid_geometry import Grid2D
from .dfsutil import _valid_item_numbers, _valid_timesteps, _get_item_info


def write_dfs2(filename: str, ds: Dataset, title="") -> None:
    dfs = _write_dfs2_header(filename, ds, title)
    _write_dfs2_data(dfs, ds)


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
    proj = geometry.projection_string
    origin = geometry.origin
    orient = geometry.orientation

    if geometry.is_geo:
        proj = factory.CreateProjectionGeoOrigin(proj, *origin, orient)
    else:
        cart: Cartography = Cartography.CreateProjOrigin(proj, *origin, orient)
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


def _write_dfs2_data(dfs: DfsFile, ds: Dataset) -> None:

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

            d = d.reshape(ds.shape[-2:])  # spatial axes
            darray = d.flatten()

            if not ds.is_equidistant:
                t_rel = (ds.time[i] - ds.time[0]).total_seconds()

            dfs.WriteItemTimeStepNext(t_rel, darray.astype(np.float32))

    dfs.Close()


class Dfs2(_Dfs123):

    _ndim = 2

    def __init__(self, filename=None, type="horizontal"):
        super(Dfs2, self).__init__(filename)

        self._dx = None
        self._dy = None
        self._nx = None
        self._ny = None
        self._x0 = 0
        self._y0 = 0
        self.geometry = None

        if filename:
            self._read_dfs2_header()
            is_spectral = type == "spectral"

            if self._projstr == "LONG/LAT":
                if np.abs(self._orientation) < 1e-6:
                    origin = self._longitude, self._latitude
                    self.geometry = Grid2D(
                        dx=self._dx,
                        dy=self._dy,
                        nx=self._nx,
                        ny=self._ny,
                        x0=self._x0,
                        y0=self._y0,
                        origin=origin,
                        projection=self._projstr,
                        is_spectral=is_spectral,
                    )
                else:
                    raise ValueError(
                        "LONG/LAT with non-zero orientation is not supported"
                    )
            else:
                lon, lat = self._longitude, self._latitude
                cart = Cartography.CreateGeoOrigin(
                    projectionString=self._projstr,
                    lonOrigin=self._longitude,
                    latOrigin=self._latitude,
                    orientation=self._orientation,
                )
                origin_projected = np.round(cart.Geo2Proj(lon, lat), 7)
                orientation_projected = cart.OrientationProj

                self.geometry = Grid2D(
                    dx=self._dx,
                    dy=self._dy,
                    nx=self._nx,
                    ny=self._ny,
                    x0=self._x0,
                    y0=self._y0,
                    orientation=orientation_projected,
                    origin=origin_projected,
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

    def _read_dfs2_header(self):
        if not os.path.isfile(self._filename):
            raise Exception(f"file {self._filename} does not exist!")

        self._dfs = DfsFileFactory.Dfs2FileOpen(self._filename)
        self._source = self._dfs
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
        time_steps=None,
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
        if time_steps is not None:
            warnings.warn(
                FutureWarning(
                    "time_steps have been renamed to time, and will be removed in a future release"
                )
            )
            time = time_steps

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
        return Dataset(
            data_list, time=time, items=items, geometry=geometry, validate=False
        )

    def find_nearest_elements(
        self,
        lon,
        lat,
    ):
        warnings.warn(
            "find_nearest_elements is deprecated, use .geometry.find_index instead",
            FutureWarning,
        )
        projection = self._dfs.FileInfo.Projection
        axis = self._dfs.SpatialAxis
        cart = Cartography(
            projection.WKTString,
            projection.Longitude,
            projection.Latitude,
            projection.Orientation,
        )

        xx, yy = cart.Geo2Xy(lon, lat)

        j = int(xx / axis.Dx + 0.5)
        k = axis.YCount - int(yy / axis.Dy + 0.5) - 1

        j = min(max(0, j), axis.XCount - 1)
        k = min(max(0, k), axis.YCount - 1)

        return k, j

    def _open(self):
        self._dfs = DfsFileFactory.Dfs2FileOpen(self._filename)
        self._source = self._dfs

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
        coordinate=None,
        title=None,
        keep_open=False,
    ):
        """
        Create a dfs2 file

        Parameters
        ----------

        filename: str
            Location to write the dfs2 file
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
        coordinate:
            list of [projection, origin_x, origin_y, orientation]
            e.g. ['LONG/LAT', 12.4387, 55.2257, 327]
        title: str, optional
            title of the dfs2 file. Default is blank.
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

        self._write(
            filename,
            data,
            start_time,
            dt,
            datetimes,
            items,
            coordinate,
            title,
            keep_open,
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

    def plot(
        self,
        z,
        *,
        title=None,
        label=None,
        cmap=None,
        figsize=None,
        ax=None,
    ):

        warnings.warn(
            "Dfs2.plot() is deprecated, use DataArray.plot() instead",
            FutureWarning,
        )

        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        if len(z) == 1:  # if single-item Dataset
            z = z[0].to_numpy().copy()

            if z.shape[0] == 1:
                z = np.squeeze(z).copy()  # handles single time step

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        if z.ndim != 2:
            raise ValueError(
                "Only 2d data is supported. Hint select a specific timestep: e.g. z[0]"
            )

        if cmap is None:
            cmap = cm.viridis

        if self.is_geo and self.orientation == 0.0:
            lats = [self.latitude + self.dy * i for i in range(self.ny)]
            lons = [self.longitude + self.dx * i for i in range(self.nx)]

            cf = ax.imshow(z, extent=(lons[0], lons[-1], lats[0], lats[-1]), cmap=cmap)

        else:
            # TODO get spatial axes in this case as well
            cf = ax.imshow(z)

        fig.colorbar(cf, ax=ax, label=label)

        if title:
            ax.set_title(title)

        return ax
