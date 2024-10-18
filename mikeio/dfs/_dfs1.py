from __future__ import annotations
from pathlib import Path

from mikecore.DfsFactory import DfsBuilder, DfsFactory
from mikecore.DfsFile import DfsFile, DfsSimpleType
from mikecore.DfsFileFactory import DfsFileFactory
from mikecore.eum import eumQuantity, eumUnit

from .. import __dfs_version__
from ..dataset import Dataset
from ._dfs import (
    _Dfs123,
    write_dfs_data,
)
from ..eum import TimeStepUnit
from ..spatial import Grid1D


def write_dfs1(filename: str | Path, ds: Dataset, title: str = "") -> None:
    dfs = _write_dfs1_header(filename, ds, title)
    write_dfs_data(dfs=dfs, ds=ds, n_spatial_dims=1)


def _write_dfs1_header(filename: str | Path, ds: Dataset, title: str) -> DfsFile:
    builder = DfsBuilder.Create(title, "mikeio", __dfs_version__)
    builder.SetDataType(0)

    geometry: Grid1D = ds.geometry

    factory = DfsFactory()
    proj = factory.CreateProjectionGeoOrigin(
        ds.geometry.projection_string,
        ds.geometry.origin[0],
        ds.geometry.origin[1],
        ds.geometry.orientation,
    )
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
    time_axis = factory.CreateTemporalEqCalendarAxis(timestep_unit, ds.time[0], 0, dt)
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

    def __init__(self, filename: str | Path) -> None:
        super().__init__(filename)

        self._dfs = DfsFileFactory.Dfs1FileOpen(str(filename))
        self._x0: float = self._dfs.SpatialAxis.X0
        self._dx: float = self._dfs.SpatialAxis.Dx
        self._nx: int = self._dfs.SpatialAxis.XCount

        origin = self._longitude, self._latitude
        self._geometry = Grid1D(
            x0=self._x0,
            dx=self._dx,
            nx=self._nx,
            projection=self._projstr,
            origin=origin,
            orientation=self._orientation,
        )

    def _open(self) -> None:
        self._dfs = DfsFileFactory.Dfs1FileOpen(self._filename)

    @property
    def geometry(self) -> Grid1D:
        assert isinstance(self._geometry, Grid1D)
        return self._geometry

    @property
    def x0(self) -> float:
        """Start point of x values (often 0)."""
        return self._x0

    @property
    def dx(self) -> float:
        """Step size in x direction."""
        return self._dx

    @property
    def nx(self) -> int:
        """Number of node values."""
        return self._nx
