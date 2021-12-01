import os
import numpy as np
import warnings
from mikecore.eum import eumUnit
from mikecore.DfsFileFactory import DfsFileFactory
from mikecore.DfsFactory import DfsBuilder
from mikecore.Projections import Cartography

from .dfs import _Dfs123


class Dfs2(_Dfs123):

    _ndim = 2

    def __init__(self, filename=None):
        super(Dfs2, self).__init__(filename)

        self._dx = None
        self._dy = None
        self._nx = None
        self._ny = None
        self._x0 = 0
        self._y0 = 0

        if filename:
            self._read_dfs2_header()

    def __repr__(self):
        out = ["<mikeio.Dfs2>"]

        if os.path.isfile(self._filename):
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
        self._x0 = self._dfs.SpatialAxis.X0
        self._y0 = self._dfs.SpatialAxis.Y0
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
        keep_open: bool, optional
            Keep file open for appending
        """
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
        """
        Plot dfs2 data

        Parameters
        ----------

        z: np.array (2d)
        title: str, optional
            axes title
        label: str, optional
            colorbar label (or title if contour plot)
        cmap: matplotlib.cm.cmap, optional
            colormap, default viridis
        figsize: (float, float), optional
            specify size of figure
        ax: matplotlib.axes, optional
            Adding to existing axis, instead of creating new fig

        Returns
        -------
        <matplotlib.axes>

        Examples
        --------
        >>> dfs = Dfs2("data/gebco_sound.dfs2")
        >>> ds = dfs.read()
        >>> elevation = ds['Elevation']
        >>> dfs.plot(elevation[0], cmap='jet')
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        if len(z) == 1:  # if single-item Dataset
            z = z[0].copy()

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
        Reprojection is only available in mikeio==0.6.3
        """

        raise NotImplementedError("Reprojection is no longer available in mikeio")
