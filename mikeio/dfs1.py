import os

from mikecore.DfsBuilder import DfsBuilder
from mikecore.DfsFileFactory import DfsFileFactory
from mikecore.eum import eumUnit

from . import __dfs_version__
from .dfs import _Dfs123
from .spatial.grid_geometry import Grid1D


class Dfs1(_Dfs123):
    _ndim = 1

    def __init__(self, filename=None):
        super().__init__(filename)

        self._dx = None
        self._nx = None
        self._x0 = 0

        if filename:
            self._read_dfs1_header()
            origin = self._longitude, self._latitude
            self.geometry = Grid1D(
                x0=self._x0,
                dx=self._dx,
                nx=self._nx,
                projection=self._projstr,
                origin=origin,
                orientation=self._orientation,
            )

    def __repr__(self):
        out = ["<mikeio.Dfs1>"]

        if os.path.isfile(self._filename):
            out.append(f"dx: {self.dx:.5f}")

            if self._n_items is not None:
                if self._n_items < 10:
                    out.append("items:")
                    for i, item in enumerate(self.items):
                        out.append(f"  {i}:  {item}")
                else:
                    out.append(f"number of items: {self._n_items}")
            if os.path.isfile(self._filename):
                if self._n_timesteps == 1:
                    out.append("time: time-invariant file (1 step)")
                else:
                    out.append(f"time: {self._n_timesteps} steps")
                    out.append(f"start time: {self._start_time}")

        return str.join("\n", out)

    def _read_dfs1_header(self):
        if not os.path.isfile(self._filename):
            raise FileNotFoundError(self._filename)

        self._dfs = DfsFileFactory.Dfs1FileOpen(self._filename)
        self._x0 = self._dfs.SpatialAxis.X0
        self._dx = self._dfs.SpatialAxis.Dx
        self._nx = self._dfs.SpatialAxis.XCount

        self._read_header()

    def _open(self):
        self._dfs = DfsFileFactory.Dfs1FileOpen(self._filename)

    def write(
        self,
        filename,
        data,
        dt=None,
        dx=1,
        x0=0,
        title=None,
    ):
        """
        Write a dfs1 file

        Parameters
        ----------
        filename: str
            Location to write the dfs1 file
        data: Dataset
            list of matrices, one for each item. Matrix dimension: x, time
        dt: float
            The time step in seconds.
        x0:
            Lower right position
        dx:
            length of each grid in the x direction (meters)
        title: str, optional
            title of the dfs file (can be blank)

        """

        self._x0 = x0

        if isinstance(data, list):
            raise TypeError(
                "supplying data as a list of numpy arrays is deprecated, please supply data in the form of a Dataset"
            )

        self._builder = DfsBuilder.Create(title, "mikeio", __dfs_version__)
        self._dx = dx
        self._write(filename=filename, data=data, dt=dt, title=title)

    def _set_spatial_axis(self):
        self._builder.SetSpatialAxis(
            self._factory.CreateAxisEqD1(
                eumUnit.eumUmeter, self._nx, self._x0, self._dx
            )
        )

    @property
    def x0(self):
        """Start point of x values (often 0)"""
        return self._x0

    @property
    def dx(self):
        """Step size in x direction"""
        return self._dx

    @property
    def nx(self):
        """Number of node values"""
        return self._nx
