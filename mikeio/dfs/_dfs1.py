from pathlib import Path

from mikecore.DfsFactory import DfsBuilder, DfsFactory
from mikecore.DfsFile import DfsFile, DfsSimpleType
from mikecore.DfsFileFactory import DfsFileFactory
from mikecore.eum import eumQuantity, eumUnit

from .. import __dfs_version__
from ..dataset import Dataset
from ._dfs import (
    _Dfs123,
    _write_dfs_data,
)
from ..eum import TimeStepUnit
from ..spatial import Grid1D


def write_dfs1(filename: str | Path, ds: Dataset, title="") -> None:
    dfs = _write_dfs1_header(filename, ds, title)
    _write_dfs_data(dfs=dfs, ds=ds, n_spatial_dims=1)


def _write_dfs1_header(filename: str | Path, ds: Dataset, title="") -> DfsFile:
    builder = DfsBuilder.Create(title, "mikeio", __dfs_version__)
    builder.SetDataType(0)

    geometry: Grid1D = ds.geometry

    factory = DfsFactory()
    proj = factory.CreateProjectionGeoOrigin(ds.geometry.projection_string,ds.geometry.origin[0],ds.geometry.origin[1],ds.geometry.orientation)
    builder.SetGeographicalProjection(proj)
    builder.SetSpatialAxis(
        factory.CreateAxisEqD1(
            eumUnit.eumUmeter,
            geometry._nx,
            geometry._x0,
            geometry._dx,
        )
    )

    timestep_unit = TimeStepUnit.SECOND
    dt = ds.timestep or 1.0  # It can not be None
    time_axis = factory.CreateTemporalEqCalendarAxis(
        timestep_unit, ds.time[0], 0, dt
    )
    builder.SetTemporalAxis(time_axis)

    for item in ds.items:
        builder.AddCreateDynamicItem(
            item.name,
            eumQuantity.Create(item.type, item.unit),
            DfsSimpleType.Float,
            item.data_value_type,
        )

    try:
        builder.CreateFile(str(filename))
    except IOError:
        print("cannot create dfs file: ", filename)

    return builder.GetFile()
    

class Dfs1(_Dfs123):
    _ndim = 1

    def __init__(self, filename):
        super().__init__(filename)
        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(path)

        self._dfs = DfsFileFactory.Dfs1FileOpen(str(filename))
        self._x0 = self._dfs.SpatialAxis.X0
        self._dx = self._dfs.SpatialAxis.Dx
        self._nx = self._dfs.SpatialAxis.XCount

        self._read_header()

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

        out.append(f"dx: {self.dx:.5f}")

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
    

    def _open(self):
        self._dfs = DfsFileFactory.Dfs1FileOpen(self._filename)

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
