from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection, Sequence, Tuple

import numpy as np
import pandas as pd
from mikecore.DfsFactory import DfsFactory
from mikecore.DfsuBuilder import DfsuBuilder
from mikecore.DfsuFile import DfsuFile, DfsuFileType
from mikecore.eum import eumQuantity, eumUnit
from tqdm import trange

from mikeio.spatial._utils import xy_to_bbox

from .. import __dfs_version__
from ..dataset import Dataset
from ..dfs._dfs import (
    _get_item_info,
    _read_item_time_step,
    _valid_item_numbers,
    _valid_timesteps,
)
from ..spatial import (
    GeometryFM2D,
)
from ..spatial import Grid2D
from .._track import _extract_track
from ._common import get_elements_from_source, get_nodes_from_source
from ..eum import ItemInfo


def write_dfsu(filename: str | Path, data: Dataset) -> None:
    """Write a dfsu file

    Parameters
    ----------
    filename: str
        dfsu filename
    data: Dataset
        Dataset to be written
    """
    filename = str(filename)

    if len(data.time) == 1:
        dt = 1  # TODO is there any sensible default?
    else:
        if not data.is_equidistant:
            raise ValueError("Non-equidistant time axis is not supported.")

        dt = (data.time[1] - data.time[0]).total_seconds()  # type: ignore
    n_time_steps = len(data.time)

    geometry = data.geometry
    dfsu_filetype = DfsuFileType.Dfsu2D

    if geometry.is_layered:
        dfsu_filetype = geometry._type.value

    xn = geometry.node_coordinates[:, 0]
    yn = geometry.node_coordinates[:, 1]
    zn = geometry.node_coordinates[:, 2]

    elem_table = [np.array(e) + 1 for e in geometry.element_table]

    builder = DfsuBuilder.Create(dfsu_filetype)
    if dfsu_filetype != DfsuFileType.Dfsu2D:
        builder.SetNumberOfSigmaLayers(geometry.n_sigma_layers)

    builder.SetNodes(xn, yn, zn, geometry.codes)
    builder.SetElements(elem_table)

    factory = DfsFactory()
    proj = factory.CreateProjection(geometry.projection_string)
    builder.SetProjection(proj)
    builder.SetTimeInfo(data.time[0], dt)
    builder.SetZUnit(eumUnit.eumUmeter)

    if dfsu_filetype != DfsuFileType.Dfsu2D:
        builder.SetNumberOfSigmaLayers(geometry.n_sigma_layers)

    for item in data.items:
        builder.AddDynamicItem(item.name, eumQuantity.Create(item.type, item.unit))

    builder.ApplicationTitle = "mikeio"
    builder.ApplicationVersion = __dfs_version__
    dfs = builder.CreateFile(filename)

    for i in range(n_time_steps):
        if geometry.is_layered:
            if "time" in data.dims:
                assert data._zn is not None
                zn = data._zn[i]
            else:
                zn = data._zn
            dfs.WriteItemTimeStepNext(0, zn.astype(np.float32))
        for da in data:
            if "time" in data.dims:
                d = da.to_numpy()[i, :]
            else:
                d = da.to_numpy()
            d[np.isnan(d)] = data.deletevalue
            dfs.WriteItemTimeStepNext(0, d.astype(np.float32))
    dfs.Close()


def _validate_elements_and_geometry_sel(elements: Any, **kwargs: Any) -> None:
    """Check that only one of elements, area, x, y is selected"""
    used_kwargs = [key for key, val in kwargs.items() if val is not None]

    if elements is not None and len(used_kwargs) > 0:
        raise ValueError(f"Cannot select both {used_kwargs} and elements!")

    if "area" in used_kwargs and ("x" in used_kwargs or "y" in used_kwargs):
        raise ValueError("Cannot select both x,y and area!")


@dataclass
class _DfsuInfo:
    filename: str
    type: DfsuFileType
    time: pd.DatetimeIndex
    timestep: float
    items: list[ItemInfo]
    deletevalue: float


def _get_dfsu_info(filename: str | Path) -> _DfsuInfo:
    filename = str(filename)
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"file {path} does not exist!")
    dfs = DfsuFile.Open(filename)
    type = DfsuFileType(dfs.DfsuFileType)
    deletevalue = dfs.DeleteValueFloat
    freq = pd.Timedelta(seconds=dfs.TimeStepInSeconds)
    time = pd.date_range(
        start=dfs.StartDateTime,
        periods=dfs.NumberOfTimeSteps,
        freq=freq,
    )
    timestep = dfs.TimeStepInSeconds
    items = _get_item_info(dfs.ItemInfo)
    dfs.Close()
    return _DfsuInfo(
        filename=filename,
        type=type,
        time=time,
        timestep=timestep,
        items=items,
        deletevalue=deletevalue,
    )


# class _Dfsu:
#     show_progress = False

#     def __init__(self, filename: str | Path) -> None:
#         """
#         Create a Dfsu object

#         Parameters
#         ---------
#         filename: str
#             dfsu filename
#         """
#         info = _get_dfsu_info(filename)
#         self._filename = info.filename
#         self._type = info.type
#         self._deletevalue = info.deletevalue
#         self._time = info.time
#         self._timestep = info.timestep
#         self._items = info.items

#     def __repr__(self):
#         out = [f"<mikeio.{self.__class__.__name__}>"]

#         if self._type is not DfsuFileType.DfsuSpectral0D:
#             if self._type is not DfsuFileType.DfsuSpectral1D:
#                 out.append(f"number of elements: {self.geometry.n_elements}")
#             out.append(f"number of nodes: {self.geometry.n_nodes}")
#         if self.geometry.is_spectral:
#             if self.geometry.n_directions > 0:
#                 out.append(f"number of directions: {self.geometry.n_directions}")
#             if self.geometry.n_frequencies > 0:
#                 out.append(f"number of frequencies: {self.geometry.n_frequencies}")
#         if self.geometry.projection_string:
#             out.append(f"projection: {self.geometry.projection_string}")
#         if self.geometry.is_layered:
#             out.append(f"number of sigma layers: {self.geometry.n_sigma_layers}")
#         if (
#             self._type == DfsuFileType.DfsuVerticalProfileSigmaZ
#             or self._type == DfsuFileType.Dfsu3DSigmaZ
#         ):
#             out.append(
#                 f"max number of z layers: {self.geometry.n_layers - self.geometry.n_sigma_layers}"
#             )
#         if self.n_items < 10:
#             out.append("items:")
#             for i, item in enumerate(self.items):
#                 out.append(f"  {i}:  {item}")
#         else:
#             out.append(f"number of items: {self.geometry.n_items}")
#         if self.n_timesteps == 1:
#             out.append(f"time: time-invariant file (1 step) at {self.time[0]}")
#         else:
#             out.append(
#                 f"time: {str(self.time[0])} - {str(self.time[-1])} ({self.n_timesteps} records)"
#             )
#         return str.join("\n", out)

#     def _read_items(self, filename: str) -> list[ItemInfo]:
#         dfs = DfsuFile.Open(filename)
#         items = _get_item_info(dfs.ItemInfo)
#         dfs.Close()
#         return items

#     @property
#     def geometry(self):
#         return self._geometry

#     @property
#     def deletevalue(self) -> float:
#         """File delete value"""
#         return self._deletevalue

#     @property
#     def n_items(self) -> int:
#         """Number of items"""
#         return len(self.items)

#     @property
#     def items(self) -> list[ItemInfo]:
#         """List of items"""
#         return self._items

#     @property
#     def start_time(self) -> pd.Timestamp:
#         """File start time"""
#         return self._time[0]

#     @property
#     def n_timesteps(self) -> int:
#         """Number of time steps"""
#         return len(self._time)

#     @property
#     def timestep(self) -> float:
#         """Time step size in seconds"""
#         return self._timestep

#     @property
#     def end_time(self) -> pd.Timestamp:
#         """File end time"""
#         return self._time[-1]

#     @property
#     def time(self) -> pd.DatetimeIndex:
#        return self._time


class Dfsu2DH:
    show_progress = False

    def __init__(self, filename: str | Path) -> None:
        info = _get_dfsu_info(filename)
        self._filename = info.filename
        self._type = info.type
        self._deletevalue = info.deletevalue
        self._time = info.time
        self._timestep = info.timestep
        self._items = info.items
        self._geometry = self._read_geometry(self._filename)

    def __repr__(self):
        out = [f"<mikeio.{self.__class__.__name__}>"]

        out.append(f"number of elements: {self.geometry.n_elements}")
        out.append(f"number of nodes: {self.geometry.n_nodes}")
        out.append(f"projection: {self.geometry.projection_string}")
        if self.n_items < 10:
            out.append("items:")
            for i, item in enumerate(self.items):
                out.append(f"  {i}:  {item}")
        else:
            out.append(f"number of items: {self.geometry.n_items}")
        if self.n_timesteps == 1:
            out.append(f"time: time-invariant file (1 step) at {self.time[0]}")
        else:
            out.append(
                f"time: {str(self.time[0])} - {str(self.time[-1])} ({self.n_timesteps} records)"
            )
        return str.join("\n", out)

    @property
    def geometry(self):
        return self._geometry

    @property
    def deletevalue(self) -> float:
        """File delete value"""
        return self._deletevalue

    @property
    def n_items(self) -> int:
        """Number of items"""
        return len(self.items)

    @property
    def items(self) -> list[ItemInfo]:
        """List of items"""
        return self._items

    @property
    def start_time(self) -> pd.Timestamp:
        """File start time"""
        return self._time[0]

    @property
    def n_timesteps(self) -> int:
        """Number of time steps"""
        return len(self._time)

    @property
    def timestep(self) -> float:
        """Time step size in seconds"""
        return self._timestep

    @property
    def end_time(self) -> pd.Timestamp:
        """File end time"""
        return self._time[-1]

    @property
    def time(self) -> pd.DatetimeIndex:
        return self._time

    @staticmethod
    def _read_geometry(filename: str) -> GeometryFM2D:
        dfs = DfsuFile.Open(filename)
        dfsu_type = DfsuFileType(dfs.DfsuFileType)

        node_table = get_nodes_from_source(dfs)
        el_table = get_elements_from_source(dfs)

        geometry = GeometryFM2D(
            node_coordinates=node_table.coordinates,
            element_table=el_table.connectivity,
            codes=node_table.codes,
            projection=dfs.Projection.WKTString,
            dfsu_type=dfsu_type,
            element_ids=el_table.ids,
            node_ids=node_table.ids,
            validate=False,
        )
        dfs.Close()
        return geometry

    def read(
        self,
        *,
        items: str | int | Sequence[str | int] | None = None,
        time: int | str | slice | None = None,
        elements: Collection[int] | None = None,
        area: Tuple[float, float, float, float] | None = None,
        x: float | None = None,
        y: float | None = None,
        keepdims: bool = False,
        dtype: Any = np.float32,
        error_bad_data: bool = True,
        fill_bad_data_value: float = np.nan,
    ) -> Dataset:
        """
        Read data from a dfsu file

        Parameters
        ---------
        items: list[int] or list[str], optional
            Read only selected items, by number (0-based), or by name
        time: int, str, datetime, pd.TimeStamp, sequence, slice or pd.DatetimeIndex, optional
            Read only selected time steps, by default None (=all)
        keepdims: bool, optional
            When reading a single time step only, should the time-dimension be kept
            in the returned Dataset? by default: False
        area: list[float], optional
            Read only data inside (horizontal) area given as a
            bounding box (tuple with left, lower, right, upper)
            or as list of coordinates for a polygon, by default None
        x, y: float, optional
            Read only data for elements containing the (x,y) points(s),
            by default None
        elements: list[int], optional
            Read only selected element ids, by default None
        error_bad_data: bool, optional
            raise error if data is corrupt, by default True,
        fill_bad_data_value:
            fill value for to impute corrupt data, used in conjunction with error_bad_data=False
            default np.nan

        Returns
        -------
        Dataset
            A Dataset with data dimensions [t,elements]
        """

        if dtype not in [np.float32, np.float64]:
            raise ValueError("Invalid data type. Choose np.float32 or np.float64")
        dfs = DfsuFile.Open(self._filename)

        single_time_selected, time_steps = _valid_timesteps(dfs, time)

        _validate_elements_and_geometry_sel(elements, area=area, x=x, y=y)
        if elements is None:
            elements = self._parse_geometry_sel(area=area, x=x, y=y)

        if elements is None:
            geometry = self.geometry
            n_elems = geometry.n_elements
        else:
            elements = [elements] if np.isscalar(elements) else list(elements)  # type: ignore
            n_elems = len(elements)
            geometry = self.geometry.elements_to_geometry(elements)

        item_numbers = _valid_item_numbers(dfs.ItemInfo, items)
        items = _get_item_info(dfs.ItemInfo, item_numbers)
        n_items = len(item_numbers)

        deletevalue = self.deletevalue

        data_list = []

        shape: Tuple[int, ...]

        n_steps = len(time_steps)
        shape = (
            (n_elems,)
            if (single_time_selected and not keepdims)
            else (n_steps, n_elems)
        )
        for item in range(n_items):
            # Initialize an empty data block
            data: np.ndarray = np.ndarray(shape=shape, dtype=dtype)
            data_list.append(data)

        time = self.time

        for i in trange(n_steps, disable=not self.show_progress):
            it = time_steps[i]
            for item in range(n_items):
                dfs, d = _read_item_time_step(
                    dfs=dfs,
                    filename=self._filename,
                    time=time,
                    item_numbers=item_numbers,
                    deletevalue=deletevalue,
                    shape=shape,
                    item=item,
                    it=it,
                    error_bad_data=error_bad_data,
                    fill_bad_data_value=fill_bad_data_value,
                )

                if elements is not None:
                    d = d[elements]

                if single_time_selected and not keepdims:
                    data_list[item] = d
                else:
                    data_list[item][i] = d

        time = self.time[time_steps]

        dfs.Close()

        dims: Tuple[str, ...]

        dims = ("time", "element")

        if single_time_selected and not keepdims:
            dims = ("element",)

        if elements is not None and len(elements) == 1:
            # squeeze point data
            dims = tuple([d for d in dims if d != "element"])
            data_list = [np.squeeze(d, axis=-1) for d in data_list]

        return Dataset(
            data_list, time, items, geometry=geometry, dims=dims, validate=False
        )

    def _parse_geometry_sel(self, area, x, y):
        """Parse geometry selection

        Parameters
        ----------
        area : list[float], optional
            Read only data inside (horizontal) area given as a
            bounding box (tuple with left, lower, right, upper)
            or as list of coordinates for a polygon, by default None
        x : float, optional
            Read only data for elements containing the (x,y) points(s),
            by default None
        y : float, optional
            Read only data for elements containing the (x,y) points(s),
            by default None

        Returns
        -------
        list[int]
            List of element ids

        Raises
        ------
        ValueError
            If no elements are found in selection
        """
        elements = None

        if area is not None:
            elements = self.geometry._elements_in_area(area)

        if (x is not None) or (y is not None):
            elements = self.geometry.find_index(x=x, y=y)

        if (x is not None) or (y is not None) or (area is not None):
            # selection was attempted
            if (elements is None) or len(elements) == 0:
                raise ValueError("No elements in selection!")

        return elements

    def get_overset_grid(self, dx=None, dy=None, nx=None, ny=None, buffer=0.0):
        """get a 2d grid that covers the domain by specifying spacing or shape

        Parameters
        ----------
        dx : float, optional
            grid resolution in x-direction (or in x- and y-direction)
        dy : float, optional
            grid resolution in y-direction
        nx : int, optional
            number of points in x-direction,
            by default None (the value will be inferred)
        ny : int, optional
            number of points in y-direction,
            by default None (the value will be inferred)
        buffer : float, optional
            positive to make the area larger, default=0
            can be set to a small negative value to avoid NaN
            values all around the domain.

        Returns
        -------
        <mikeio.Grid2D>
            2d grid
        """
        nc = self.geometry.geometry2d.node_coordinates
        bbox = xy_to_bbox(nc, buffer=buffer)
        return Grid2D(
            bbox=bbox,
            dx=dx,
            dy=dy,
            nx=nx,
            ny=ny,
            projection=self.geometry.projection_string,
        )

    def _dfs_read_item_time_func(
        self, item: int, step: int
    ) -> Tuple[np.ndarray, pd.Timestamp]:
        dfs = DfsuFile.Open(self._filename)
        itemdata = dfs.ReadItemTimeStep(item + 1, step)

        return itemdata.Data, itemdata.Time

    def extract_track(self, track, items=None, method="nearest", dtype=np.float32):
        """
        Extract track data from a dfsu file

        Parameters
        ---------
        track: pandas.DataFrame
            with DatetimeIndex and (x, y) of track points as first two columns
            x,y coordinates must be in same coordinate system as dfsu
        track: str
            filename of csv or dfs0 file containing t,x,y
        items: list[int] or list[str], optional
            Extract only selected items, by number (0-based), or by name
        method: str, optional
            Spatial interpolation method ('nearest' or 'inverse_distance')
            default='nearest'

        Returns
        -------
        Dataset
            A dataset with data dimension t
            The first two items will be x- and y- coordinates of track

        Examples
        --------
        >>> dfsu = mikeio.open("tests/testdata/NorthSea_HD_and_windspeed.dfsu")
        >>> ds = dfsu.extract_track("tests/testdata/altimetry_NorthSea_20171027.csv")
        >>> ds
        <mikeio.Dataset>
        dims: (time:1115)
        time: 2017-10-26 04:37:37 - 2017-10-30 20:54:47 (1115 non-equidistant records)
        geometry: GeometryUndefined()
        items:
          0:  Longitude <Undefined> (undefined)
          1:  Latitude <Undefined> (undefined)
          2:  Surface elevation <Surface Elevation> (meter)
          3:  Wind speed <Wind speed> (meter per sec)
        """
        dfs = DfsuFile.Open(self._filename)

        item_numbers = _valid_item_numbers(dfs.ItemInfo, items)
        items = _get_item_info(dfs.ItemInfo, item_numbers)
        _, time_steps = _valid_timesteps(dfs, time_steps=None)

        res = _extract_track(
            deletevalue=self.deletevalue,
            start_time=self.start_time,
            end_time=self.end_time,
            timestep=self.timestep,
            geometry=self.geometry,
            n_elements=self.geometry.n_elements,
            track=track,
            items=items,
            time_steps=time_steps,
            item_numbers=item_numbers,
            method=method,
            dtype=dtype,
            data_read_func=self._dfs_read_item_time_func,
        )
        dfs.Close()
        return res
