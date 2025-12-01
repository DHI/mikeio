from __future__ import annotations
from collections.abc import Iterable
from pathlib import Path
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence
import numpy as np
import pandas as pd

from mikecore.DfsFile import (
    DfsDynamicItemInfo,
    DfsFile,
    DfsFileInfo,
    TimeAxisType,
)
from mikecore.DfsFileFactory import DfsFileFactory
from mikecore.Projections import Cartography

from ..dataset import Dataset
from ..eum import ItemInfo, ItemInfoList
from ..exceptions import ItemsError
from .._time import DateTimeSelector


@dataclass
class DfsHeader:
    n_items: int
    n_timesteps: int
    start_time: datetime
    dt: float
    coordinates: tuple[str, float, float, float]
    items: list[ItemInfo]


def _read_item_time_step(
    *,
    dfs: DfsFile,
    filename: str,
    time: pd.DatetimeIndex,
    item_numbers: list[int],
    deletevalue: float,
    shape: tuple[int, ...],
    item: int,
    it: int,
    error_bad_data: bool = True,
    fill_bad_data_value: float = np.nan,
) -> tuple[DfsFile, np.ndarray, float]:
    itemdata = dfs.ReadItemTimeStep(item_numbers[item] + 1, it)
    t = itemdata.Time
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
    return dfs, d, t


def _fuzzy_item_search(
    *, dfsItemInfo: list[DfsDynamicItemInfo], search: str, start_idx: int = 0
) -> list[int]:
    import fnmatch

    names = [info.Name for info in dfsItemInfo]
    item_numbers = [
        i - start_idx for i, name in enumerate(names) if fnmatch.fnmatch(name, search)
    ]
    if len(item_numbers) == 0:
        raise KeyError(f"No items like: {search} found. Valid names are {names}")
    return item_numbers


def _valid_item_numbers(
    dfsItemInfo: list[DfsDynamicItemInfo],
    items: str | int | Sequence[int | str] | None = None,
    ignore_first: bool = False,
) -> list[int]:
    start_idx = 1 if ignore_first else 0
    n_items_file = len(dfsItemInfo) - start_idx
    if items is None:
        return list(range(n_items_file))

    # Handling scalar and sequences is a bit tricky

    item_numbers: list[int] = []

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


def _valid_timesteps(
    dfsFileInfo: DfsFileInfo, time_steps: int | Sequence[int] | str | slice | None
) -> tuple[bool, list[int]]:
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

        if time_step_file <= 0:
            if nt > 1:
                raise ValueError(
                    f"Time step must be a positive number. Time step in the file is {time_step_file} seconds."
                )

            warnings.warn(
                f"Time step is {time_step_file} seconds. This must be a positive number. Setting to 1 second."
            )
            time_step_file = 1

        freq = pd.Timedelta(seconds=time_step_file)
        time = pd.date_range(start_time_file, periods=nt, freq=freq)
    elif time_axis.TimeAxisType == TimeAxisType.CalendarNonEquidistant:
        idx = list(range(nt))
        if time_steps is None:
            return False, idx

        if isinstance(time_steps, int):
            return True, [idx[time_steps]]

        if isinstance(time_steps, Iterable):
            # check that all elements are integers and are in the range of nt
            if not all(isinstance(i, int) for i in time_steps):
                raise ValueError("All elements in time_steps must be integers.")
            if not all(0 <= i < nt for i in time_steps):  # type: ignore
                raise ValueError(
                    f"All elements in time_steps must be in the range of 0 to {nt-1}."
                )
            return False, list(time_steps)  # type: ignore

        raise TypeError(
            f"Temporal selection with type: {type(time_steps)} is not supported"
        )

    dts = DateTimeSelector(time)

    idx = dts.isel(time_steps)

    if isinstance(time_steps, str):
        if len(idx) == 1:
            single_time_selected = True

    return single_time_selected, idx


def _item_numbers_by_name(
    dfsItemInfo: list[DfsDynamicItemInfo],
    item_names: list[str],
    ignore_first: bool = False,
) -> list[int]:
    """Utility function to find item numbers.

    Parameters
    ----------
    dfsItemInfo : list[DfsDynamicItemInfo]
        item info from dfs file
    item_names : list[str]
        Names of items to be found
    ignore_first : bool, optional
        Ignore first item, by default False


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
    dfsItemInfo: list[DfsDynamicItemInfo],
    item_numbers: list[int] | None = None,
    ignore_first: bool = False,
) -> ItemInfoList:
    """Read DFS ItemInfo for specific item numbers.

    Parameters
    ----------
    dfsItemInfo : list[DfsDynamicItemInfo]
        Item info from dfs file
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


def write_dfs_data(*, dfs: DfsFile, ds: Dataset, n_spatial_dims: int) -> None:
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

    def __init__(self, filename: str | Path) -> None:
        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(path)
        self._filename = str(filename) if filename else None
        self._end_time = None
        self._is_equidistant = True
        dfs = DfsFileFactory.DfsGenericOpen(self._filename)
        self._dfs = dfs
        self._geometry: Any = None  # Handled by subclass
        n_items = len(dfs.ItemInfo)
        self._items = self._get_item_info(list(range(n_items)))
        if dfs.FileInfo.TimeAxis.TimeAxisType in {
            TimeAxisType.CalendarEquidistant,
            TimeAxisType.CalendarNonEquidistant,
        }:
            self._start_time = dfs.FileInfo.TimeAxis.StartDateTime
        else:  # relative time axis
            self._start_time = datetime(
                1970, 1, 1
            )  # TODO is this the proper epoch, should this magic number be somewhere else?

        if hasattr(dfs.FileInfo.TimeAxis, "TimeStep"):
            self._timestep = (
                # some files have dt = 0 😳
                dfs.FileInfo.TimeAxis.TimeStepInSeconds()
                if dfs.FileInfo.TimeAxis.TimeStepInSeconds() > 0
                else 1
            )  # TODO handle other timeunits

            freq = pd.Timedelta(seconds=self._timestep)
            self._time = pd.date_range(
                start=self._start_time,
                periods=dfs.FileInfo.TimeAxis.NumberOfTimeSteps,
                freq=freq,
            )
        else:
            self._timestep = None
            self._time = None
        self._n_timesteps: int = dfs.FileInfo.TimeAxis.NumberOfTimeSteps
        projstr = dfs.FileInfo.Projection.WKTString
        self._projstr: str = "NON-UTM" if not projstr else projstr
        self._longitude: float = dfs.FileInfo.Projection.Longitude
        self._latitude: float = dfs.FileInfo.Projection.Latitude
        self._orientation: float = dfs.FileInfo.Projection.Orientation
        self._deletevalue: float = dfs.FileInfo.DeleteValueFloat

        dfs.Close()

    def __repr__(self) -> str:
        name = self.__class__.__name__
        out = [f"<mikeio.{name}>"]

        out.append(f"geometry: {self.geometry}")
        if self.n_items < 10:
            out.append("items:")
            for i, item in enumerate(self.items):
                out.append(f"  {i}:  {item}")
        else:
            out.append(f"number of items: {self.n_items}")

        if self._n_timesteps == 1:
            out.append("time: time-invariant file (1 step)")
        else:
            out.append(f"time: {self._n_timesteps} steps")
            out.append(f"start time: {self._start_time}")

        return str.join("\n", out)

    def _open(self) -> None:
        raise NotImplementedError("Should be implemented by subclass")

    def _get_item_info(self, item_numbers: Sequence[int]) -> list[ItemInfo]:
        """Read DFS ItemInfo.

        Parameters
        ----------
        item_numbers : list[int]
            Item numbers to read

        Returns
        -------
        list[Iteminfo]

        """
        infos = self._dfs.ItemInfo
        nos = item_numbers
        return [ItemInfo.from_mikecore_dynamic_item_info(infos[i]) for i in nos]

    @property
    def geometry(self) -> Any:
        return self._geometry

    @property
    def deletevalue(self) -> float:
        "File delete value."
        return self._deletevalue

    @property
    def n_items(self) -> int:
        "Number of items."
        return len(self.items)

    @property
    def items(self) -> list[ItemInfo]:
        "List of items."
        return self._items

    @property
    def time(self) -> pd.DatetimeIndex | None:
        return self._time

    @property
    def start_time(self) -> pd.Timestamp:
        """File start time."""
        return self._start_time

    @property
    def end_time(self) -> pd.Timestamp:
        """File end time."""
        if self._end_time is None:
            self._end_time = self.read(items=[0]).time[-1].to_pydatetime()  # type: ignore

        return self._end_time

    @property
    def n_timesteps(self) -> int:
        """Number of time steps."""
        return self._n_timesteps

    @property
    def timestep(self) -> Any:
        """Time step size in seconds."""
        # this will fail if the TimeAxisType is not calendar and equidistant, but that is ok
        return self._dfs.FileInfo.TimeAxis.TimeStepInSeconds()

    @property
    def projection_string(self) -> str:
        return self._projstr

    @property
    def longitude(self) -> float:
        """Origin longitude."""
        return self._longitude

    @property
    def latitude(self) -> float:
        """Origin latitude."""
        return self._latitude

    @property
    def origin(self) -> Any:
        """Origin (in own CRS)."""
        return self.geometry.origin

    @property
    def orientation(self) -> Any:
        """Orientation (in own CRS)."""
        return self.geometry.orientation

    @property
    def is_geo(self) -> bool:
        """Are coordinates geographical (LONG/LAT)?"""
        return self._projstr == "LONG/LAT"

    def _validate_no_orientation_in_geo(self) -> None:
        if self.is_geo and abs(self._orientation) > 1e-6:
            raise ValueError("Orientation is not supported for LONG/LAT coordinates")

    def _origin_and_orientation_in_CRS(self) -> tuple[Any, float]:
        """Project origin and orientation to projected CRS (if not LONG/LAT)."""
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
            origin = tuple(np.round(cart.Geo2Proj(lon, lat), 6))  # type: ignore
            orientation = cart.OrientationProj
        return origin, orientation
