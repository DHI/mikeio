import os

from DHI.Generic.MikeZero import eumUnit
from DHI.Generic.MikeZero.DFS import DfsFileFactory
from DHI.Generic.MikeZero.DFS.dfs123 import Dfs1Builder

from .dfs import AbstractDfs


class Dfs1(AbstractDfs):
    _ndim = 1
    _dx = None
    _nx = None
    _x0 = 0

    def __init__(self, filename=None):
        super(Dfs1, self).__init__(filename)

        if filename:
            self._read_dfs1_header()

    def __repr__(self):
        out = ["Dfs1"]

        if self._filename:
            out.append(f"dx: {self.dx:.5f}")

            if self._n_items is not None:
                if self._n_items < 10:
                    out.append("Items:")
                    for i, item in enumerate(self.items):
                        out.append(f"  {i}:  {item}")
                else:
                    out.append(f"Number of items: {self._n_items}")
            if self._filename:
                if self._n_timesteps == 1:
                    out.append("Time: time-invariant file (1 step)")
                else:
                    out.append(f"Time: {self._n_timesteps} steps")
                    out.append(f"Start time: {self._start_time}")

        return str.join("\n", out)

    def _read_dfs1_header(self):
        if not os.path.isfile(self._filename):
            raise FileNotFoundError(self._filename)

        self._dfs = DfsFileFactory.Dfs1FileOpen(self._filename)
        self._dx = self._dfs.SpatialAxis.Dx
        self._nx = self._dfs.SpatialAxis.XCount

        self._read_header()

    def _open(self):
        self._dfs = DfsFileFactory.Dfs1FileOpen(self._filename)

    def write(
        self,
        filename,
        data,
        start_time=None,
        dt=None,
        items=None,
        dx=1,
        x0=0,
        coordinate=None,
        title=None,
    ):
        """
        Write a dfs1 file

        Parameters
        ----------
        filename: str
            Location to write the dfs1 file
        data: list[np.array]
            list of matrices, one for each item. Matrix dimension: x, time
        start_time: datetime, optional
            start datetime
        dt: float
            The time step in seconds.
        items: list[ItemInfo], optional
            List of ItemInfo (e.g. Water Level).
        coordinate:
            list of [projection, origin_x, origin_y, orientation]
            e.g. ['LONG/LAT', 12.4387, 55.2257, 327]
        x0:
            Lower right position
        dx:
            length of each grid in the x direction (meters)
        title: str, optional
            title of the dfs file (can be blank)

        """

        self._builder = Dfs1Builder.Create(title, "mikeio", 0)
        self._dx = dx
        self._write(filename, data, start_time, dt, items, coordinate, title)

    def _set_spatial_axis(self):
        self._builder.SetSpatialAxis(
            self._factory.CreateAxisEqD1(
                eumUnit.eumUmeter, self._nx, self._x0, self._dx
            )
        )

    @property
    def dx(self):
        """Step size in x direction
        """
        return self._dx
