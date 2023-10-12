from __future__ import annotations
import warnings
from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple, Sequence
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

from mikecore.DfsFactory import DfsFactory
from mikecore.DfsFile import (
    DfsDynamicItemInfo,
    DfsFile,
    DfsFileInfo,
    DfsSimpleType,
    TimeAxisType,
)
from mikecore.DfsFileFactory import DfsFileFactory
from mikecore.eum import eumQuantity
from mikecore.Projections import Cartography

from ..dataset import Dataset
from ..eum import EUMType, EUMUnit, ItemInfo, ItemInfoList, TimeStepUnit
from ..exceptions import DataDimensionMismatch, ItemsError
from ..spatial import GeometryUndefined
from .._time import DateTimeSelector

@dataclass
class DfsHeader:

    n_items: int
    n_timesteps: int
    start_time: datetime
    dt: float
    coordinates: Tuple[str, float, float, float]
    items: List[ItemInfo]


def _read_item_time_step(
    *,
    dfs,
    filename,
    time,
    item_numbers,
    deletevalue,
    shape,
    item,
    it,
    error_bad_data=True,
    fill_bad_data_value=np.nan,
):
    itemdata = dfs.ReadItemTimeStep(item_numbers[item] + 1, it)
    if itemdata is not None:
        d = itemdata.Data
        d[d == deletevalue] = np.nan
    else:
        if error_bad_data:
            raise ValueError(f"Error reading: {time[it]}")
        else:
            warnings.warn(f"Error reading: {time[it]}")
            d = np.zeros(shape[1])
            d[:] = fill_bad_data_value
            dfs.Close()
            dfs = DfsFileFactory.DfsGenericOpen(filename)
    return dfs, d


def _fuzzy_item_search(
    *, dfsItemInfo: List[DfsDynamicItemInfo], search: str, start_idx: int = 0
):
    import fnmatch

    names = [info.Name for info in dfsItemInfo]
    item_numbers = [
        i - start_idx for i, name in enumerate(names) if fnmatch.fnmatch(name, search)
    ]
    if len(item_numbers) == 0:
        raise KeyError(f"No items like: {search} found. Valid names are {names}")
    return item_numbers


def _valid_item_numbers(
    dfsItemInfo: List[DfsDynamicItemInfo],
    items: Optional[str | int | List[int] | List[str]] = None,
    ignore_first: bool = False,
) -> List[int]:
    start_idx = 1 if ignore_first else 0
    n_items_file = len(dfsItemInfo) - start_idx
    if items is None:
        return list(range(n_items_file))

    # Handling scalar and sequences is a bit tricky

    item_numbers: List[int] = []

    # check if items is a scalar (int or str)
    if isinstance(items, (int, str)):
        if isinstance(items, str) and "*" in items:
            return _fuzzy_item_search(
                dfsItemInfo=dfsItemInfo, search=items, start_idx=start_idx
            )
        elif isinstance(items, str):
            item_number = _item_numbers_by_name(dfsItemInfo, [items], ignore_first)[0]
            return [item_number]
        elif isinstance(items, int):
            if (items < 0) or (items >= n_items_file):
                raise ItemsError(n_items_file)
            return [items]

    assert isinstance(items, Sequence)
    for item in items:
        if isinstance(item, str):
            item_number = _item_numbers_by_name(dfsItemInfo, [item], ignore_first)[0]
        elif isinstance(item, int):
            if (item < 0) or (item >= n_items_file):
                raise ItemsError(n_items_file)
            item_number = item
        else:
            raise ItemsError(n_items_file)
        item_numbers.append(item_number)

    if len(set(item_numbers)) != len(item_numbers):
        raise ValueError("'items' must be unique")

    return item_numbers


def _valid_timesteps(dfsFileInfo: DfsFileInfo, time_steps) -> Tuple[bool, List[int]]:

    time_axis = dfsFileInfo.TimeAxis

    single_time_selected = False
    if isinstance(time_steps, (int, datetime)):
        single_time_selected = True

    nt = time_axis.NumberOfTimeSteps

    if time_axis.TimeAxisType != TimeAxisType.CalendarEquidistant:
        # TODO is this the proper epoch, should this magic number be somewhere else?
        start_time_file = datetime(1970, 1, 1)
    else:
        start_time_file = time_axis.StartDateTime

    if time_axis.TimeAxisType in (
        TimeAxisType.CalendarEquidistant,
        TimeAxisType.TimeEquidistant,
    ):
        time_step_file = time_axis.TimeStep
        freq = pd.Timedelta(seconds=time_step_file)
        time = pd.date_range(start_time_file, periods=nt, freq=freq)
    elif time_axis.TimeAxisType == TimeAxisType.CalendarNonEquidistant:
        idx = list(range(nt))

        if isinstance(time_steps, int):
            return True, [idx[time_steps]]
        return single_time_selected, idx

    dts = DateTimeSelector(time)

    idx = dts.isel(time_steps)

    if isinstance(time_steps, str):
        if len(idx) == 1:
            single_time_selected = True

    return single_time_selected, idx


def _item_numbers_by_name(
    dfsItemInfo, item_names: List[str], ignore_first: bool = False
) -> List[int]:
    """Utility function to find item numbers

    Parameters
    ----------
    dfsItemInfo : MIKE dfs ItemInfo object

    item_names : list[str]
        Names of items to be found

    Returns
    -------
    list[int]
        item numbers (0-based)

    Raises
    ------
    KeyError
        In case item is not found in the dfs file
    """
    first_idx = 1 if ignore_first else 0
    names = [x.Name for x in dfsItemInfo[first_idx:]]

    item_lookup = {name: i for i, name in enumerate(names)}
    try:
        item_numbers = [item_lookup[x] for x in item_names]
    except KeyError:
        raise KeyError(f"Selected item name not found. Valid names are {names}")

    return item_numbers


def _get_item_info(
    dfsItemInfo: List[DfsDynamicItemInfo],
    item_numbers: Optional[List[int]] = None,
    ignore_first: bool = False,
) -> ItemInfoList:
    """Read DFS ItemInfo for specific item numbers

    Parameters
    ----------
    dfsItemInfo : List[DfsDynamicItemInfo]
    item_numbers : list[int], optional
        Item numbers to read, by default all items are read
    ignore_first : bool, optional
        Ignore first item, by default False, used for Dfsu3D

    Returns
    -------
    ItemInfoList
    """
    first_idx = 1 if ignore_first else 0
    if item_numbers is None:
        item_numbers = list(range(len(dfsItemInfo) - first_idx))

    item_numbers = [i + first_idx for i in item_numbers]
    items = [
        ItemInfo.from_mikecore_dynamic_item_info(dfsItemInfo[i]) for i in item_numbers
    ]
    return ItemInfoList(items)


def _write_dfs_data(*, dfs: DfsFile, ds: Dataset, n_spatial_dims: int) -> None:

    deletevalue = dfs.FileInfo.DeleteValueFloat  # ds.deletevalue
    has_no_time = "time" not in ds.dims
    if ds.is_equidistant:
        t_rel = np.zeros(ds.n_timesteps)
    else:
        t_rel = (ds.time - ds.time[0]).total_seconds()

    for i in range(ds.n_timesteps):
        for item in range(ds.n_items):

            if has_no_time:
                d = ds[item].values
            else:
                d = ds[item].values[i]
            d = d.copy()  # to avoid modifying the input
            d[np.isnan(d)] = deletevalue

            d = d.reshape(ds.shape[-n_spatial_dims:])  # spatial axes
            darray = d.flatten()

            dfs.WriteItemTimeStepNext(t_rel[i], darray.astype(np.float32))

    dfs.Close()


class _Dfs123:
    _ndim: int

    show_progress = False

    # TODO add all common arguments
    def __init__(self, filename=None):
        self._filename = str(filename) if filename else None
        self._projstr = None
        self._start_time = None
        self._end_time = None
        self._is_equidistant = True
        self._items = None
        self._builder = None
        self._factory = None
        self._deletevalue = None
        self._override_coordinates = False
        self._timeseries_unit = TimeStepUnit.SECOND
        self._dt = None
        self.geometry = GeometryUndefined()
        self._dfs = None
        self._source = None

    def read(
        self,
        *,
        items=None,
        time=None,
        keepdims=False,
        dtype=np.float32,
    ) -> Dataset:
        """
        Read data from a dfs file

        Parameters
        ---------
        items: list[int] or list[str], optional
            Read only selected items, by number (0-based), or by name
        time: int, str, datetime, pd.TimeStamp, sequence, slice or pd.DatetimeIndex, optional
            Read only selected time steps, by default None (=all)
        keepdims: bool, optional
            When reading a single time step only, should the time-dimension be kept
            in the returned Dataset? by default: False

        Returns
        -------
        Dataset
        """

        self._open()

        item_numbers = _valid_item_numbers(self._dfs.ItemInfo, items)
        n_items = len(item_numbers)

        single_time_selected, time_steps = _valid_timesteps(self._dfs.FileInfo, time)
        nt = len(time_steps) if not single_time_selected else 1

        shape: Tuple[int, ...]

        if self._ndim == 1:
            shape = (nt, self._nx)
        elif self._ndim == 2:
            shape = (nt, self._ny, self._nx)
        else:
            shape = (nt, self._nz, self._ny, self._nx)

        if single_time_selected and not keepdims:
            shape = shape[1:]

        data_list: List[np.ndarray] = [
            np.ndarray(shape=shape, dtype=dtype) for _ in range(n_items)
        ]

        t_seconds = np.zeros(len(time_steps))

        for i, it in enumerate(tqdm(time_steps, disable=not self.show_progress)):
            for item in range(n_items):

                itemdata = self._dfs.ReadItemTimeStep(item_numbers[item] + 1, int(it))

                src = itemdata.Data
                d = src

                d[d == self.deletevalue] = np.nan

                if self._ndim == 2:
                    d = d.reshape(self._ny, self._nx)

                if single_time_selected:
                    data_list[item] = d
                else:
                    data_list[item][i] = d

            t_seconds[i] = itemdata.Time

        time = pd.to_datetime(t_seconds, unit="s", origin=self.start_time)  # type: ignore

        items = _get_item_info(self._dfs.ItemInfo, item_numbers)

        self._dfs.Close()
        return Dataset(data_list, time, items, geometry=self.geometry, validate=False)

    def _read_header(self):
        dfs = self._dfs
        self._n_items = len(dfs.ItemInfo)
        self._items = self._get_item_info(list(range(self._n_items)))
        self._timeaxistype = dfs.FileInfo.TimeAxis.TimeAxisType
        if self._timeaxistype in {
            TimeAxisType.CalendarEquidistant,
            TimeAxisType.CalendarNonEquidistant,
        }:
            self._start_time = dfs.FileInfo.TimeAxis.StartDateTime
        else:  # relative time axis
            self._start_time = datetime(
                1970, 1, 1
            )  # TODO is this the proper epoch, should this magic number be somewhere else?
        if hasattr(dfs.FileInfo.TimeAxis, "TimeStep"):
            self._timestep_in_seconds = (
                dfs.FileInfo.TimeAxis.TimeStep
            )  # TODO handle other timeunits
            # TODO to get the EndTime
        self._n_timesteps = dfs.FileInfo.TimeAxis.NumberOfTimeSteps
        projstr = dfs.FileInfo.Projection.WKTString
        self._projstr = "NON-UTM" if not projstr else projstr
        self._longitude = dfs.FileInfo.Projection.Longitude
        self._latitude = dfs.FileInfo.Projection.Latitude
        self._orientation = dfs.FileInfo.Projection.Orientation
        self._deletevalue = dfs.FileInfo.DeleteValueFloat

        dfs.Close()

    def _write(
        self,
        *,
        filename,
        ds,
        dt,
        coordinate=None,
        title,
        keep_open=False,
    ):
        
        assert isinstance(ds, Dataset)

        neq_datetimes = None
        if isinstance(ds, Dataset) and not ds.is_equidistant:
            neq_datetimes = ds.time

        header, data = self._write_handle_common_arguments(
            title=title, data=ds, dt=dt, coordinate=coordinate
        )

        
        shape = np.shape(data[0])
        t_offset = 0 if len(shape) == self._ndim else 1

        # TODO find out a clever way to handle the grid dimensions
        if self._ndim == 1:
            self._nx = shape[t_offset + 0]
        elif self._ndim == 2:
            self._ny = shape[t_offset + 0]
            self._nx = shape[t_offset + 1]
        elif self._ndim == 3:
            self._nz = shape[t_offset + 0]
            self._ny = shape[t_offset + 1]
            self._nx = shape[t_offset + 2]

        self._factory = DfsFactory()

        # TODO pass grid
        self._set_spatial_axis()

        if self._ndim == 1:
            if not all(np.shape(d)[t_offset + 0] == self._nx for d in data):
                raise DataDimensionMismatch()

        if self._ndim == 2:
            if not all(np.shape(d)[t_offset + 0] == self._ny for d in data):
                raise DataDimensionMismatch()

            if not all(np.shape(d)[t_offset + 1] == self._nx for d in data):
                raise DataDimensionMismatch()

        if neq_datetimes is not None:
            self._is_equidistant = False
            start_time = neq_datetimes[0]
            self._start_time = start_time

        dfs = self._setup_header(filename, header)
        self._dfs = dfs

        deletevalue = dfs.FileInfo.DeleteValueFloat  # -1.0000000031710769e-30

        for i in trange(header.n_timesteps, disable=not self.show_progress):
            for item in range(header.n_items):

                d = data[item][i] if t_offset == 1 else data[item]
                d = d.copy()  # to avoid modifying the input
                d[np.isnan(d)] = deletevalue

                if self._is_equidistant:
                    dfs.WriteItemTimeStepNext(0, d.astype(np.float32))
                else:
                    t = neq_datetimes[i]
                    relt = (t - self._start_time).total_seconds()
                    dfs.WriteItemTimeStepNext(relt, d.astype(np.float32))

        if not keep_open:
            dfs.Close()
        else:
            return self

    def append(self, data: Dataset) -> None:
        
        warnings.warn(FutureWarning("append() is deprecated."))

        if not data.dims == ("time", "y", "x"):
            raise NotImplementedError(
                    "Append is only available for 2D files with dims ('time', 'y', 'x')"
                )

        deletevalue = self._dfs.FileInfo.DeleteValueFloat  # -1.0000000031710769e-30

        for i in trange(data.n_timesteps, disable=not self.show_progress):
            for da in data:

                values = da.to_numpy()
                d = values[i]
                d = d.copy()  # to avoid modifying the input
                d[np.isnan(d)] = deletevalue

                d = d.reshape(data.shape[1:])
                darray = d.reshape(d.size, 1)[:, 0]
                self._dfs.WriteItemTimeStepNext(0, darray.astype(np.float32))
                
                   
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self._dfs.Close()

    def close(self):
        "Finalize write for a dfs file opened with `write(...,keep_open=True)`"
        self._dfs.Close()

    def _write_handle_common_arguments(self, *, title: Optional[str], data: Dataset, coordinate, dt: Optional[float] = None):

        if title is None:
            self._title = ""

        n_timesteps = data.n_timesteps
        n_items = data.n_items

        if coordinate is None:
            if self._projstr is not None:
                coordinate = [
                    self._projstr,
                    self._longitude,
                    self._latitude,
                    self._orientation,
                ]
            elif isinstance(data, Dataset) and (data.geometry is not None):
                coordinate = [
                    data.geometry.projection_string,
                    data.geometry.origin[0],
                    data.geometry.origin[1],
                    data.geometry.orientation,
                ]
            else:
                warnings.warn("No coordinate system provided")
                coordinate = ["LONG/LAT", 0, 0, 0]
        else:
            self._override_coordinates = True

        assert isinstance(data, Dataset), "data must be supplied in the form of a mikeio.Dataset"

        items = data.items
        start_time = data.time[0]
        n_timesteps = len(data.time)
        if dt is None and len(data.time) > 1:
            dt = (data.time[1] - data.time[0]).total_seconds()
        data = data.to_numpy()

        if dt is None:
            dt = 1
            if n_timesteps > 1:
                warnings.warn("No timestep supplied. Using 1s.")

        if items is None:
            items = [ItemInfo(f"Item {i+1}") for i in range(self._n_items)]

        header = DfsHeader(n_items=n_items, n_timesteps=n_timesteps, dt=dt, start_time=start_time, coordinates=coordinate, items=items)
        return header, data

    def _setup_header(self, filename: str, header: DfsHeader):

        system_start_time = header.start_time

        self._builder.SetDataType(0)

        proj = self._factory.CreateProjectionGeoOrigin(*header.coordinates)

        self._builder.SetGeographicalProjection(proj)

        if self._is_equidistant:
            self._builder.SetTemporalAxis(
                self._factory.CreateTemporalEqCalendarAxis(
                    self._timeseries_unit, system_start_time, 0, header.dt
                )
            )
        else:
            self._builder.SetTemporalAxis(
                self._factory.CreateTemporalNonEqCalendarAxis(
                    self._timeseries_unit, system_start_time
                )
            )

        for item in header.items:
            self._builder.AddCreateDynamicItem(
                item.name,
                eumQuantity.Create(item.type, item.unit),
                DfsSimpleType.Float,
                item.data_value_type,
            )

        try:
            self._builder.CreateFile(filename)
        except IOError:
            # TODO does this make sense?
            print("cannot create dfs file: ", filename)

        return self._builder.GetFile()

    def _open(self):
        raise NotImplementedError("Should be implemented by subclass")

    def _set_spatial_axis(self):
        raise NotImplementedError("Should be implemented by subclass")

    def _find_item(self, item_names):
        """Utility function to find item numbers

        Parameters
        ----------
        dfs : DfsFile

        item_names : list[str]
            Names of items to be found

        Returns
        -------
        list[int]
            item numbers (0-based)

        Raises
        ------
        KeyError
            In case item is not found in the dfs file
        """
        names = [x.Name for x in self._dfs.ItemInfo]
        item_lookup = {name: i for i, name in enumerate(names)}
        try:
            item_numbers = [item_lookup[x] for x in item_names]
        except KeyError:
            raise KeyError(f"Selected item name not found. Valid names are {names}")

        return item_numbers

    def _get_item_info(self, item_numbers):
        """Read DFS ItemInfo

        Parameters
        ----------
        dfs : MIKE dfs object
        item_numbers : list[int]

        Returns
        -------
        list[Iteminfo]
        """
        items = []
        for item in item_numbers:
            name = self._dfs.ItemInfo[item].Name
            eumItem = self._dfs.ItemInfo[item].Quantity.Item
            eumUnit = self._dfs.ItemInfo[item].Quantity.Unit
            itemtype = EUMType(eumItem)
            unit = EUMUnit(eumUnit)
            data_value_type = self._dfs.ItemInfo[item].ValueType
            item = ItemInfo(name, itemtype, unit, data_value_type)
            items.append(item)
        return items

    @property
    def deletevalue(self):
        "File delete value"
        return self._deletevalue

    @property
    def n_items(self):
        "Number of items"
        return self._n_items

    @property
    def items(self):
        "List of items"
        return self._items

    @property
    def start_time(self):
        """File start time"""
        return self._start_time

    @property
    def end_time(self):
        """File end time"""
        if self._end_time is None:
            self._end_time = self.read(items=[0]).time[-1].to_pydatetime()

        return self._end_time

    @property
    def n_timesteps(self) -> int:
        """Number of time steps"""
        return self._n_timesteps

    @property
    def timestep(self) -> float:
        """Time step size in seconds"""
        # this will fail if the TimeAxisType is not calendar and equidistant, but that is ok
        return self._dfs.FileInfo.TimeAxis.TimeStepInSeconds()

    @property
    def time(self) -> pd.DatetimeIndex:
        """File all datetimes"""
        # this will fail if the TimeAxisType is not calendar and equidistant, but that is ok
        if not self._is_equidistant:
            raise NotImplementedError("Not implemented for non-equidistant files")
        return pd.date_range(
            start=self.start_time, periods=self.n_timesteps, freq=f"{self.timestep}S"
        )

    @property
    def projection_string(self):
        return self._projstr

    @property
    def longitude(self):
        """Origin longitude"""
        return self._longitude

    @property
    def latitude(self):
        """Origin latitude"""
        return self._latitude

    @property
    def origin(self):
        """Origin (in own CRS)"""
        return self.geometry.origin

    @property
    def orientation(self):
        """Orientation (in own CRS)"""
        return self.geometry.orientation

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        """Shape of the data array"""
        pass

    @property
    @abstractmethod
    def dx(self):
        """Step size in x direction"""
        pass

    def _validate_no_orientation_in_geo(self):
        if self.is_geo and abs(self._orientation) > 1e-6:
            raise ValueError("Orientation is not supported for LONG/LAT coordinates")

    def _origin_and_orientation_in_CRS(self):
        """Project origin and orientation to projected CRS (if not LONG/LAT)"""
        if self.is_geo:
            origin = self._longitude, self._latitude
            orientation = 0.0
        else:
            lon, lat = self._longitude, self._latitude
            cart = Cartography.CreateGeoOrigin(
                projectionString=self._projstr,
                lonOrigin=lon,
                latOrigin=lat,
                orientation=self._orientation,
            )
            # convert origin and orientation to projected CRS
            origin = tuple(np.round(cart.Geo2Proj(lon, lat), 6))
            orientation = cart.OrientationProj
        return origin, orientation
