import os
import warnings
from DHI.Generic.MikeZero import eumUnit
from DHI.Generic.MikeZero.DFS import DfsFileFactory
from DHI.Generic.MikeZero.DFS.dfs123 import Dfs2Builder, Dfs2Reprojector
from DHI.Generic.MikeZero.DFS import DfsFileFactory
from DHI.Generic.MikeZero.DFS.dfs123 import Dfs2Builder
from DHI.Projections import Cartography


from .dfs import _Dfs123


class Dfs2(_Dfs123):

    _ndim = 2
    _dx = None
    _dy = None
    _nx = None
    _ny = None
    _x0 = 0
    _y0 = 0

    def __init__(self, filename=None):
        super(Dfs2, self).__init__(filename)

        if filename:
            self._read_dfs2_header()

    def __repr__(self):
        out = ["<mikeio.Dfs2>"]

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

                if self._n_timesteps == 1:
                    out.append("Time: time-invariant file (1 step)")
                else:
                    out.append(f"Time: {self._n_timesteps} steps")
                    out.append(f"Start time: {self._start_time}")

        return str.join("\n", out)

    def _read_dfs2_header(self):
        if not os.path.isfile(self._filename):
            raise Exception(f"file {self._filename} does not exist!")

        self._dfs = DfsFileFactory.Dfs2FileOpen(self._filename)
        self._source = self._dfs
        self._dx = self._dfs.SpatialAxis.Dx
        self._dy = self._dfs.SpatialAxis.Dy
        self._nx = self._dfs.SpatialAxis.XCount
        self._ny = self._dfs.SpatialAxis.YCount
        if self._dfs.FileInfo.TimeAxis.TimeAxisType == 4:
            self._is_equidistant = False

        self._read_header()

    def find_nearest_element(self, lon, lat):
        warnings.warn("OBSOLETE! method name changed to find_nearest_elements")
        return self.find_nearest_elements(lon, lat)

    def find_nearest_elements(
        self,
        lon,
        lat,
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
        dt: datetime
            The list of datetimes for the case of nonEquadistant Timeaxis.
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
        """

        self._builder = Dfs2Builder.Create(title, "mikeio", 0)
        if not self._dx:
            self._dx = 1
        if dx:
            self._dx = dx

        if not self._dy:
            self._dy = 1
        if dy:
            self._dy = dy

        self._write(filename, data, start_time, dt, datetimes, items, coordinate, title)

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
    def dx(self):
        """Step size in x direction"""
        return self._dx

    @property
    def dy(self):
        """Step size in y direction"""
        return self._dy

    @property
    def shape(self):
        return (self._n_timesteps, self._ny, self._nx)

    def reproject(
        self,
        filename,
        projectionstring,
        dx,
        dy,
        longitude_origin=None,
        latitude_origin=None,
        nx=None,
        ny=None,
        orientation=0.0,
        interpolate=True,
    ):
        """
        Reproject and write results to a new dfs2 file

            Parameters
            ----------
            filename: str
                location to write the reprojected dfs2 file
            projectionstring: str
                WKT string of new projection
            dx: float
                length of each grid in the x direction (projection units)
            dy: float
                length of each grid in the y direction (projection units)
            latitude_origin: float, optional
                latitude at origin of new grid, default same as original
            longitude_origin: float, optional
                longitude at origin of new grid, default same as original
            nx: int, optional
                n grid points in x direction, default same as original
            ny: int, optional
                n grid points in y direction, default same as original
            orientation: float, optional
                rotated grid, default 0.0
            interpolate: bool, optional
                interpolate to new grid, default true

            Examples
            --------

            >>> dfs = Dfs2("input.dfs")
            >>> dfs.reproject("out.dfs2", projectionstring="UTM-33",
            ...               dx=200.0, dy=200.0,
            ...               longitude_origin=12.0, latitude_origin=55.0,
            ...               nx=285, ny=612)
        """

        if nx is None:
            nx = self.shape[2]

        if ny is None:
            ny = self.shape[1]

        if latitude_origin is None:
            latitude_origin = self.latitude

        if longitude_origin is None:
            longitude_origin = self.longitude

        dfs2File = DfsFileFactory.Dfs2FileOpen(self._filename)

        tool = Dfs2Reprojector(dfs2File, filename)
        tool.Interpolate = interpolate

        tool.SetTarget(
            projectionstring,
            longitude_origin,
            latitude_origin,
            orientation,
            nx,
            0.0,
            dx,
            ny,
            0.0,
            dy,
        )
        tool.Process()
