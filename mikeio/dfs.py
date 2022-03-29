from datetime import datetime
from abc import abstractmethod

import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from .dataset import Dataset
from .base import TimeSeries

from .dfsutil import _valid_item_numbers, _valid_timesteps, _get_item_info
from .eum import ItemInfo, TimeStepUnit, EUMType, EUMUnit
from .custom_exceptions import DataDimensionMismatch, ItemNumbersError
from mikecore.eum import eumQuantity
from mikecore.DfsFile import DfsSimpleType, TimeAxisType
from mikecore.DfsFactory import DfsFactory


class _Dfs123(TimeSeries):

    show_progress = False

    def __init__(self, filename=None, dtype=np.float32):
        self._filename = str(filename)
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
        self.geometry = None
        self._dtype = dtype
        self._dfs = None
        self._source = None

    def read(self, *, items=None, time=None, time_steps=None) -> Dataset:
        """
        Read data from a dfs file

        Parameters
        ---------
        items: list[int] or list[str], optional
            Read only selected items, by number (0-based), or by name
        time: str, int or list[int], optional
            Read only selected times

        Returns
        -------
        Dataset
        """
        if time_steps is not None:
            warnings.warn(
                FutureWarning(
                    "time_steps have been renamed to time, and will be removed in a future release"
                )
            )
            time = time_steps

        self._open()

        item_numbers = _valid_item_numbers(self._dfs.ItemInfo, items)
        n_items = len(item_numbers)

        time_steps = _valid_timesteps(self._dfs.FileInfo, time)
        nt = len(time_steps)
        single_time_selected = np.isscalar(time) if time is not None else False

        if self._ndim == 1:
            shape = (nt, self._nx)
        elif self._ndim == 2:
            shape = (nt, self._ny, self._nx)
        else:
            shape = (nt, self._nz, self._ny, self._nx)

        if single_time_selected:
            shape = shape[1:]

        data_list = [
            np.ndarray(shape=shape, dtype=self._dtype) for item in range(n_items)
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

        time = pd.to_datetime(t_seconds, unit="s", origin=self.start_time)

        items = _get_item_info(self._dfs.ItemInfo, item_numbers)

        self._dfs.Close()
        return Dataset(data_list, time, items, geometry=self.geometry)

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
            self._start_time = datetime(1970, 1, 1)
        if hasattr(dfs.FileInfo.TimeAxis, "TimeStep"):
            self._timestep_in_seconds = (
                dfs.FileInfo.TimeAxis.TimeStep
            )  # TODO handle other timeunits
            # TODO to get the EndTime
        self._n_timesteps = dfs.FileInfo.TimeAxis.NumberOfTimeSteps
        self._projstr = dfs.FileInfo.Projection.WKTString
        self._longitude = dfs.FileInfo.Projection.Longitude
        self._latitude = dfs.FileInfo.Projection.Latitude
        self._orientation = dfs.FileInfo.Projection.Orientation
        self._deletevalue = dfs.FileInfo.DeleteValueFloat

        dfs.Close()

    def _write(
        self,
        filename,
        data,
        start_time,
        dt,
        datetimes,
        items,
        coordinate,
        title,
        keep_open=False,
    ):

        if isinstance(data, Dataset) and not data.is_equidistant:
            datetimes = data.time

        self._write_handle_common_arguments(
            title, data, items, coordinate, start_time, dt
        )

        shape = np.shape(data[0])
        if self._ndim == 1:
            self._nx = shape[1]
        elif self._ndim == 2:
            self._ny = shape[1]
            self._nx = shape[2]

        self._factory = DfsFactory()
        self._set_spatial_axis()

        if self._ndim == 1:
            if not all(np.shape(d)[1] == self._nx for d in self._data):
                raise DataDimensionMismatch()

        if self._ndim == 2:
            if not all(np.shape(d)[1] == self._ny for d in self._data):
                raise DataDimensionMismatch()

            if not all(np.shape(d)[2] == self._nx for d in self._data):
                raise DataDimensionMismatch()
        if datetimes is not None:
            self._is_equidistant = False
            start_time = datetimes[0]
            self._start_time = start_time

        dfs = self._setup_header(filename)
        self._dfs = dfs

        deletevalue = dfs.FileInfo.DeleteValueFloat  # -1.0000000031710769e-30

        for i in trange(self._n_timesteps, disable=not self.show_progress):
            for item in range(self._n_items):

                d = self._data[item][i]
                d = d.copy()  # to avoid modifying the input
                d[np.isnan(d)] = deletevalue

                if self._ndim == 1:
                    darray = d

                if self._ndim == 2:
                    d = d.reshape(self.shape[1:])
                    d = np.flipud(d)
                    darray = d.reshape(d.size, 1)[:, 0]

                if self._is_equidistant:
                    dfs.WriteItemTimeStepNext(0, darray.astype(np.float32))
                else:
                    t = datetimes[i]
                    relt = (t - self._start_time).total_seconds()
                    dfs.WriteItemTimeStepNext(relt, darray.astype(np.float32))

        if not keep_open:
            dfs.Close()
        else:
            return self

    def append(self, data: Dataset) -> None:
        """Append to a dfs file opened with `write(...,keep_open=True)`

        Parameters
        -----------
        data: Dataset
        """

        deletevalue = self._dfs.FileInfo.DeleteValueFloat  # -1.0000000031710769e-30

        for i in trange(self._n_timesteps, disable=not self.show_progress):
            for item in range(self._n_items):

                d = data[item].to_numpy()[i]
                d = d.copy()  # to avoid modifying the input
                d[np.isnan(d)] = deletevalue

                if self._ndim == 1:
                    darray = d

                if self._ndim == 2:
                    d = d.reshape(self.shape[1:])
                    darray = d.reshape(d.size, 1)[:, 0]

                if self._is_equidistant:
                    self._dfs.WriteItemTimeStepNext(0, darray.astype(np.float32))
                else:
                    raise NotImplementedError(
                        "Append is not yet available for non-equidistant files"
                    )

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self._dfs.Close()

    def close(self):
        "Finalize write for a dfs file opened with `write(...,keep_open=True)`"
        self._dfs.Close()

    def _write_handle_common_arguments(
        self, title, data, items, coordinate, start_time, dt
    ):

        if title is None:
            self._title = ""

        self._n_timesteps = np.shape(data[0])[0]
        self._n_items = len(data)

        if coordinate is None:
            if self._projstr is not None:
                self._coordinate = [
                    self._projstr,
                    self._longitude,
                    self._latitude,
                    self._orientation,
                ]
            elif isinstance(data, Dataset) and (data.geometry is not None):
                self._coordinate = [
                    data.geometry.projection_string,
                    data.geometry.origin[0],
                    data.geometry.origin[1],
                    data.geometry.orientation,
                ]
            else:
                warnings.warn("No coordinate system provided")
                self._coordinate = ["LONG/LAT", 0, 0, 0]
        else:
            self._override_coordinates = True
            self._coordinate = coordinate

        if isinstance(data, Dataset):
            self._items = data.items
            self._start_time = data.time[0]
            if dt is None and len(data.time) > 1:
                self._dt = (data.time[1] - data.time[0]).total_seconds()
            self._data = data.to_numpy()
        else:
            self._data = data

        if start_time is None:
            if self._start_time is None:
                self._start_time = datetime.now()
                warnings.warn(
                    f"No start time supplied. Using current time: {self._start_time} as start time."
                )
            else:
                self._start_time = self._start_time
        else:
            self._start_time = start_time

        if dt:
            self._dt = dt

        if self._dt is None:
            self._dt = 1
            if self._n_timesteps > 1:
                warnings.warn("No timestep supplied. Using 1s.")

        if items:
            self._items = items

        if self._items is None:
            self._items = [ItemInfo(f"Item {i+1}") for i in range(self._n_items)]

        self._timeseries_unit = TimeStepUnit.SECOND

    def _setup_header(self, filename):

        system_start_time = self._start_time

        self._builder.SetDataType(0)

        proj = self._factory.CreateProjectionGeoOrigin(*self._coordinate)

        self._builder.SetGeographicalProjection(proj)

        if self._is_equidistant:
            self._builder.SetTemporalAxis(
                self._factory.CreateTemporalEqCalendarAxis(
                    self._timeseries_unit, system_start_time, 0, self._dt
                )
            )
        else:
            self._builder.SetTemporalAxis(
                self._factory.CreateTemporalNonEqCalendarAxis(
                    self._timeseries_unit, system_start_time
                )
            )

        for item in self._items:
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

    def _validate_item_numbers(self, item_numbers):
        if not all(
            isinstance(item_number, int) and 0 <= item_number < self.n_items
            for item_number in item_numbers
        ):
            raise ItemNumbersError()

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
    def n_timesteps(self):
        """Number of time steps"""
        return self._n_timesteps

    @property
    def timestep(self):
        """Time step size in seconds"""
        return self._timestep_in_seconds

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
    def orientation(self):
        """North to Y orientation"""
        return self._orientation

    @property
    @abstractmethod
    def dx(self):
        """Step size in x direction"""
        pass
