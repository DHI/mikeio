from __future__ import annotations
from pathlib import Path
from typing import Any, Sequence

from mikecore.DfsFactory import DfsBuilder, DfsFactory
from mikecore.DfsFile import DfsFile, DfsSimpleType
from mikecore.DfsFileFactory import DfsFileFactory
from mikecore.eum import eumQuantity, eumUnit
import numpy as np
import pandas as pd
from tqdm import tqdm

from .. import __dfs_version__
from ..dataset import Dataset
from ._dfs import (
    _Dfs123,
    _get_item_info,
    _valid_item_numbers,
    _valid_timesteps,
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
    """Class for reading/writing dfs1 files.

    Parameters
    ----------
    filename:
        Path to dfs1 file

    """

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

    def read(
        self,
        *,
        items: str | int | Sequence[str | int] | None = None,
        time: int | str | slice | Sequence[int] |  None = None,
        keepdims: bool = False,
        dtype: Any = np.float32,
    ) -> Dataset:
        """Read data from a dfs1 file.

        Parameters
        ---------
        items: list[int] or list[str], optional
            Read only selected items, by number (0-based), or by name
        time: int, str, datetime, pd.TimeStamp, sequence, slice or pd.DatetimeIndex, optional
            Read only selected time steps, by default None (=all)
        keepdims: bool, optional
            When reading a single time step only, should the time-dimension be kept
            in the returned Dataset? by default: False
        dtype: data-type, optional
            Define the dtype of the returned dataset (default = np.float32)

        Returns
        -------
        Dataset

        """
        self._open()

        item_numbers = _valid_item_numbers(self._dfs.ItemInfo, items)
        n_items = len(item_numbers)

        single_time_selected, time_steps = _valid_timesteps(self._dfs.FileInfo, time)
        nt = len(time_steps) if not single_time_selected else 1
        shape: tuple[int, ...] = (nt, self.nx)
        dims = self.geometry.default_dims

        if single_time_selected and not keepdims:
            shape = shape[1:]
        else:
            dims = ("time", *dims)

        data_list: list[np.ndarray] = [
            np.ndarray(shape=shape, dtype=dtype) for _ in range(n_items)
        ]

        t_seconds = np.zeros(len(time_steps))

        for i, it in enumerate(tqdm(time_steps, disable=not self.show_progress)):
            for item in range(n_items):
                itemdata = self._dfs.ReadItemTimeStep(item_numbers[item] + 1, int(it))

                d = itemdata.Data
                assert d.ndim == 1

                d[d == self.deletevalue] = np.nan

                if single_time_selected:
                    data_list[item] = np.atleast_2d(d) if keepdims else d
                else:
                    data_list[item][i] = d

            t_seconds[i] = itemdata.Time

        time = pd.to_datetime(t_seconds, unit="s", origin=self.start_time)

        items = _get_item_info(self._dfs.ItemInfo, item_numbers)

        self._dfs.Close()

        return Dataset.from_numpy(
            data=data_list,
            time=time,
            items=items,
            dims=tuple(dims),
            geometry=self.geometry,
            validate=False,
            dt=self._timestep,
        )

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
