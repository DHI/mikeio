from __future__ import annotations
from pathlib import Path
import warnings
from functools import wraps
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


class _Dfsu:
    show_progress = False

    def __init__(self, filename: str | Path) -> None:
        """
        Create a Dfsu object

        Parameters
        ---------
        filename: str
            dfsu filename
        """
        filename = str(filename)
        self._filename = filename
        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(f"file {path} does not exist!")

        dfs = DfsuFile.Open(filename)
        self._type = DfsuFileType(dfs.DfsuFileType)
        self._deletevalue = dfs.DeleteValueFloat

        freq = pd.Timedelta(seconds=dfs.TimeStepInSeconds)

        self._time = pd.date_range(
            start=dfs.StartDateTime,
            periods=dfs.NumberOfTimeSteps,
            freq=freq,
        )
        self._timestep = dfs.TimeStepInSeconds
        dfs.Close()
        self._items = self._read_items(filename)

    def __repr__(self):
        out = []
        type_name = "Flexible Mesh" if self._type is None else self.type_name
        out.append(type_name)

        if self._type is not DfsuFileType.DfsuSpectral0D:
            if self._type is not DfsuFileType.DfsuSpectral1D:
                out.append(f"number of elements: {self.n_elements}")
            out.append(f"number of nodes: {self.n_nodes}")
        if self.is_spectral:
            if self.n_directions > 0:
                out.append(f"number of directions: {self.n_directions}")
            if self.n_frequencies > 0:
                out.append(f"number of frequencies: {self.n_frequencies}")
        if self.geometry.projection_string:
            out.append(f"projection: {self.projection_string}")
        if self.geometry.is_layered:
            out.append(f"number of sigma layers: {self.n_sigma_layers}")
        if (
            self._type == DfsuFileType.DfsuVerticalProfileSigmaZ
            or self._type == DfsuFileType.Dfsu3DSigmaZ
        ):
            out.append(f"max number of z layers: {self.n_layers - self.n_sigma_layers}")
        if hasattr(self, "items") and self.items is not None:
            if self.n_items < 10:
                out.append("items:")
                for i, item in enumerate(self.items):
                    out.append(f"  {i}:  {item}")
            else:
                out.append(f"number of items: {self.n_items}")
        if self.n_timesteps is not None:
            if self.n_timesteps == 1:
                out.append(f"time: time-invariant file (1 step) at {self._start_time}")
            else:
                out.append(
                    f"time: {str(self.time[0])} - {str(self.time[-1])} ({self.n_timesteps} records)"
                )
                out.append(f"      {self.start_time} -- {self.end_time}")
        return str.join("\n", out)

    def _read_items(self, filename: str) -> list[ItemInfo]:
        dfs = DfsuFile.Open(filename)
        items = _get_item_info(dfs.ItemInfo)
        dfs.Close()
        return items

    @property
    def type_name(self):
        """Type name, e.g. Mesh, Dfsu2D"""
        return self._type.name

    @property
    def geometry(self):
        return self._geometry

    @property
    def n_nodes(self):
        """Number of nodes"""
        return self.geometry.n_nodes

    @property
    def node_coordinates(self):
        """Coordinates (x,y,z) of all nodes"""
        return self.geometry.node_coordinates

    @property
    def node_ids(self):
        return self.geometry.node_ids

    @property
    def n_elements(self):
        """Number of elements"""
        return self.geometry.n_elements

    @property
    def element_ids(self):
        return self.geometry.element_ids

    @property
    def codes(self):
        warnings.warn(
            "property codes is deprecated, use .geometry.codes instead",
            FutureWarning,
        )
        return self.geometry.codes

    @codes.setter
    def codes(self, v):
        if len(v) != self.n_nodes:
            raise ValueError(f"codes must have length of nodes ({self.n_nodes})")
        self._geometry._codes = np.array(v, dtype=np.int32)

    @property
    def valid_codes(self):
        """Unique list of node codes"""
        return list(set(self.geometry.codes))

    @property
    def boundary_codes(self):
        """Unique list of boundary codes"""
        return [code for code in self.valid_codes if code > 0]

    @property
    def projection_string(self):
        """The projection string"""
        return self.geometry.projection_string

    @property
    def is_geo(self):
        """Are coordinates geographical (LONG/LAT)?"""
        return self.geometry.projection_string == "LONG/LAT"

    @property
    def is_local_coordinates(self):
        """Are coordinates relative (NON-UTM)?"""
        return self.geometry.projection_string == "NON-UTM"

    @property
    def element_table(self):
        """Element to node connectivity"""
        return self.geometry.element_table

    @property
    def max_nodes_per_element(self):
        """The maximum number of nodes for an element"""
        return self.geometry.max_nodes_per_element

    @property
    def is_2d(self) -> bool:
        return self._type in (
            DfsuFileType.Dfsu2D,
            DfsuFileType.DfsuSpectral2D,
        )

    @property
    def is_layered(self) -> bool:
        """Type is layered dfsu (3d, vertical profile or vertical column)"""
        return self.geometry.is_layered

    @property
    def is_spectral(self) -> bool:
        """Type is spectral dfsu (point, line or area spectrum)"""
        return self.geometry.is_spectral

    @property
    def is_tri_only(self) -> bool:
        """Does the mesh consist of triangles only?"""
        return self.geometry.is_tri_only

    @property
    def boundary_polylines(self):
        """Lists of closed polylines defining domain outline"""
        return self.geometry.boundary_polylines

    @wraps(GeometryFM2D.elements_to_geometry)
    def elements_to_geometry(self, elements, node_layers="all"):
        return self.geometry.elements_to_geometry(elements, node_layers)

    @property
    def element_coordinates(self):
        """Center coordinates of each element"""
        return self.geometry.element_coordinates

    @wraps(GeometryFM2D.contains)
    def contains(self, points):
        return self.geometry.contains(points)

    @wraps(GeometryFM2D.get_element_area)
    def get_element_area(self):
        return self.geometry.get_element_area()

    @wraps(GeometryFM2D.to_shapely)
    def to_shapely(self):
        return self.geometry.to_shapely()

    @wraps(GeometryFM2D.get_node_centered_data)
    def get_node_centered_data(self, data, extrapolate=True):
        return self.geometry.get_node_centered_data(data, extrapolate)

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

    def _read(
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
        if dtype not in [np.float32, np.float64]:
            raise ValueError("Invalid data type. Choose np.float32 or np.float64")

        # Open the dfs file for reading
        # self._read_dfsu_header(self._filename)
        dfs = DfsuFile.Open(self._filename)
        # time may have changes since we read the header
        # (if engine is continuously writing to this file)
        # TODO: add more checks that this is actually still the same file
        # (could have been replaced in the meantime)

        single_time_selected, time_steps = _valid_timesteps(dfs, time)

        self._validate_elements_and_geometry_sel(elements, area=area, x=x, y=y)
        if elements is None:
            elements = self._parse_geometry_sel(area=area, x=x, y=y)

        if elements is None:
            geometry = self.geometry
            n_elems = geometry.n_elements
        else:
            elements = [elements] if np.isscalar(elements) else list(elements)  # type: ignore
            n_elems = len(elements)
            geometry = self.geometry.elements_to_geometry(elements)

        item_numbers = _valid_item_numbers(
            dfs.ItemInfo, items, ignore_first=self.is_layered
        )
        items = _get_item_info(dfs.ItemInfo, item_numbers, ignore_first=self.is_layered)
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

    def _validate_elements_and_geometry_sel(self, elements, **kwargs):
        """Check that only one of elements, area, x, y is selected

        Parameters
        ----------
        elements : list[int], optional
            Read only selected element ids, by default None
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
        None

        Raises
        ------
        ValueError
            If more than one of elements, area, x, y is selected
        """
        used_kwargs = []
        for kw, val in kwargs.items():
            if val is not None:
                used_kwargs.append(kw)

        if elements is not None:
            for kw in used_kwargs:
                raise ValueError(f"Cannot select both {kw} and elements!")

        if "area" in used_kwargs and ("x" in used_kwargs or "y" in used_kwargs):
            raise ValueError("Cannot select both x,y and area!")

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

    def to_mesh(self, outfilename):
        """write object to mesh file

        Parameters
        ----------
        outfilename : str
            path to file to be written
        """
        self.geometry.geometry2d.to_mesh(outfilename)


class Dfsu2DH(_Dfsu):

    def __init__(self, filename: str | Path) -> None:
        super().__init__(filename)
        self._geometry = self._read_geometry(self._filename)

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

        return self._read(
            items=items,
            time=time,
            elements=elements,
            area=area,
            x=x,
            y=y,
            keepdims=keepdims,
            dtype=dtype,
            error_bad_data=error_bad_data,
            fill_bad_data_value=fill_bad_data_value,
        )

    def get_overset_grid(self, dx=None, dy=None, nx=None, ny=None, buffer=None):
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
            n_elements=self.n_elements,
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
