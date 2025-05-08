from __future__ import annotations
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from mikecore.DfsBuilder import DfsBuilder
from mikecore.DfsFactory import DfsFactory
from mikecore.DfsFile import DfsFile, DfsSimpleType
from mikecore.DfsFileFactory import DfsFileFactory
from mikecore.eum import eumQuantity, eumUnit
from mikecore.Projections import Cartography

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
from ..spatial import Grid3D


def write_dfs3(filename: str | Path, ds: Dataset, title: str = "") -> None:
    dfs = _write_dfs3_header(filename, ds, title)
    write_dfs_data(dfs=dfs, ds=ds, n_spatial_dims=3)


def _write_dfs3_header(filename: str | Path, ds: Dataset, title: str) -> DfsFile:
    builder = DfsBuilder.Create(title, "mikeio", __dfs_version__)
    builder.SetDataType(0)

    geometry: Grid3D = ds.geometry

    factory = DfsFactory()
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
    origin = geometry.origin  # Origin in geographical coordinates
    orient = geometry.orientation

    if geometry.is_geo:
        proj = factory.CreateProjectionGeoOrigin(
            geometry.projection_string, *origin, orient
        )
    else:
        cart: Cartography = Cartography.CreateProjOrigin(
            geometry.projection_string, *origin, orient
        )
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
        builder.CreateFile(str(filename))
    except IOError:
        print("cannot create dfs file: ", filename)

    return builder.GetFile()


class Dfs3(_Dfs123):
    """Class for reading/writing dfs3 files.

    Parameters
    ----------
    filename:
        Path to dfs3 file

    """

    _ndim = 3

    def __init__(self, filename: str | Path):
        super().__init__(str(filename))

        # TODO
        self._x0 = 0.0
        self._y0 = 0.0
        self._z0 = 0.0

        self._read_dfs3_header()
        self._validate_no_orientation_in_geo()
        origin, orientation = self._origin_and_orientation_in_CRS()

        self._geometry = Grid3D(
            x0=self._x0,
            dx=self._dx,
            nx=self._nx,
            y0=self._y0,
            dy=self._dy,
            ny=self._ny,
            z0=self._z0,
            dz=self._dz,
            nz=self._nz,
            origin=origin,
            projection=self._projstr,
            orientation=orientation,
        )

    def _read_dfs3_header(self, read_x0y0z0: bool = False) -> None:
        self._dfs = DfsFileFactory.Dfs3FileOpen(self._filename)

        self._source = self._dfs

        if read_x0y0z0:
            self._x0 = self._dfs.SpatialAxis.X0
            self._y0 = self._dfs.SpatialAxis.Y0
            self._z0 = self._dfs.SpatialAxis.Z0

        self._dx = self._dfs.SpatialAxis.Dx
        self._dy = self._dfs.SpatialAxis.Dy
        self._dz = self._dfs.SpatialAxis.Dz
        self._nx = self._dfs.SpatialAxis.XCount
        self._ny = self._dfs.SpatialAxis.YCount
        self._nz = self._dfs.SpatialAxis.ZCount

    def read(
        self,
        *,
        items: str | int | Sequence[str | int] | None = None,
        time: int | str | slice | Sequence[int] | None = None,
        area: tuple[float, float, float, float] | None = None,
        layers: str | int | Sequence[int] | None = None,
        keepdims: bool = False,
        dtype: Any = np.float32,
    ) -> Dataset:
        """Read data from a dfs3 file.

        Parameters
        ---------
        items: list[int] or list[str], optional
            Read only selected items, by number (0-based), or by name
        time: int, str, datetime, pd.TimeStamp, sequence, slice or pd.DatetimeIndex, optional
            Read only selected time steps, by default None (=all)
        area: tuple[float, float, float, float], optional
            Read only data within the specified rectangular area (x0, x1, y0, y1)
        keepdims: bool, optional
            When reading a single time step or a single layer only,
            should the singleton dimension be kept
            in the returned Dataset? by default: False
        layers: int, str, list[int], optional
            Read only data for specific layers, by default None
        dtype: data-type, optional
            Define the dtype of the returned dataset (default = np.float32)

        Returns
        -------
        Dataset

        """
        if area is not None:
            raise NotImplementedError("area subsetting is not yet implemented for Dfs3")
        # NOTE:
        # if keepdims is not False:
        #    return NotImplementedError("keepdims is not yet implemented for Dfs3")

        dfs = DfsFileFactory.DfsGenericOpen(self._filename)

        item_numbers = _valid_item_numbers(dfs.ItemInfo, items)
        n_items = len(item_numbers)

        single_time_selected, time_steps = _valid_timesteps(
            dfs.FileInfo, time_steps=time
        )
        nt = len(time_steps) if not single_time_selected else 1

        nz = self.geometry.nz
        ny = self.geometry.ny
        nx = self.geometry.nx
        deleteValue = dfs.FileInfo.DeleteValueFloat

        data_list = []

        if layers == "top":
            layers = -1
        layers = None if layers is None else np.atleast_1d(layers)

        dims: tuple[str, ...]
        shape: tuple[int, ...]

        nzl = nz if layers is None else len(layers)
        if nzl == 1 and (not keepdims):
            geometry = self.geometry._geometry_for_layers([0])
            dims = ("time", "y", "x")
            shape = (nt, ny, nx)
        else:
            geometry = self.geometry._geometry_for_layers(layers, keepdims)  # type: ignore
            dims = ("time", "z", "y", "x")
            shape = (nt, nzl, ny, nx)

        for item in range(n_items):
            data: np.ndarray = np.ndarray(shape=shape, dtype=dtype)
            data_list.append(data)

        if single_time_selected and not keepdims:
            shape = shape[1:]
            dims = tuple([d for d in dims if d != "time"])

        t_seconds = np.zeros(nt, dtype=float)

        for i, it in enumerate(time_steps):
            for item in range(n_items):
                itemdata = dfs.ReadItemTimeStep(item_numbers[item] + 1, int(it))
                d = itemdata.Data

                d = d.reshape(nz, ny, nx)
                d[d == deleteValue] = np.nan

                if layers is None:
                    dd = d
                elif len(layers) == 1:
                    if layers[0] == "bottom":
                        dd = self._get_bottom_values(d)
                    else:
                        dd = d[layers[0], :, :]
                else:
                    dd = d[layers, :, :]

                if single_time_selected and not keepdims:
                    data_list[item] = dd
                else:
                    data_list[item][i, ...] = dd

            t_seconds[i] = itemdata.Time

        dfs.Close()

        time = pd.to_datetime(t_seconds, unit="s", origin=self.start_time)
        items = _get_item_info(dfs.ItemInfo, item_numbers)
        return Dataset.from_numpy(
            data_list,
            time=time,
            items=items,
            geometry=geometry,
            dims=dims,
            validate=False,
        )

    def append(self, ds: Dataset, validate: bool = True) -> None:
        """Append a Dataset to an existing dfs3 file.

        Parameters
        ----------
        ds: Dataset
            Dataset to append
        validate: bool, optional
            Validate that the dataset to append has the same geometry and items as the original file

        Notes
        -----
        The original file is modified.

        """
        if validate:
            if self.geometry != ds.geometry:
                raise ValueError("The geometry of the dataset to append does not match")

            for item_s, item_o in zip(ds.items, self.items):
                if item_s != item_o:
                    raise ValueError(
                        f"Item in dataset {item_s.name} does not match {item_o.name}"
                    )

        dfs = DfsFileFactory.Dfs3FileOpenAppend(str(self._filename))
        write_dfs_data(dfs=dfs, ds=ds, n_spatial_dims=3)
        self._n_timesteps = dfs.FileInfo.TimeAxis.NumberOfTimeSteps

    @staticmethod
    def _get_bottom_values(data: np.ndarray) -> np.ndarray:
        assert len(data.shape) == 3
        b = np.empty_like(data[0])
        b[:] = np.nan
        data = np.flipud(data)
        for layer in range(data.shape[0]):  # going from surface to bottom
            y = data[layer, ...]
            b[~np.isnan(y)] = y[~np.isnan(y)]

        return b

    @property
    def geometry(self) -> Grid3D:
        return self._geometry

    @property
    def dx(self) -> float:
        """Step size in x direction."""
        return self._dx

    @property
    def dy(self) -> float:
        """Step size in y direction."""
        return self._dy

    @property
    def dz(self) -> float:
        """Step size in y direction."""
        return self._dz

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (self._n_timesteps, self._nz, self._ny, self._nx)
