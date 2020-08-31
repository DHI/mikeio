from datetime import datetime
import warnings
import numpy as np
from .helpers import safe_length
from .dutil import Dataset, get_item_info
from .dotnet import (
    to_dotnet_datetime,
    from_dotnet_datetime,
)
from .eum import ItemInfo
from DHI.Generic.MikeZero import eumQuantity
from DHI.Generic.MikeZero.DFS import (
    DfsSimpleType,
    DataValueType,
)


class Dfs123:

    _projstr = None
    _start_time = None
    _is_equidistant = True
    _items = None
    _builder = None
    _factory = None

    def __init__(self, filename=None):
        self._filename = filename

    def _read_header(self, dfs):
        self._n_items = safe_length(dfs.ItemInfo)
        self._items = get_item_info(dfs, list(range(self._n_items)))
        self._start_time = from_dotnet_datetime(dfs.FileInfo.TimeAxis.StartDateTime)
        self._n_timesteps = dfs.FileInfo.TimeAxis.NumberOfTimeSteps
        self._projstr = dfs.FileInfo.Projection.WKTString
        self._longitude = dfs.FileInfo.Projection.Longitude
        self._latitude = dfs.FileInfo.Projection.Latitude
        self._orientation = dfs.FileInfo.Projection.Orientation

        dfs.Close()

    def _write_handle_common_arguments(
        self, title, data, items, coordinate, start_time, dt
    ):
        if title is None:
            self._title = ""

        self._n_time_steps = np.shape(data[0])[0]
        self._n_items = len(data)

        if coordinate is None:
            if self._projstr is not None:
                self._coordinate = [
                    self._projstr,
                    self._longitude,
                    self._latitude,
                    self._orientation,
                ]
            else:
                warnings.warn("No coordinate system provided")
                self._coordinate = ["LONG/LAT", 0, 0, 0]
        else:
            self._coordinate = coordinate

        if isinstance(data, Dataset):
            self._items = data.items
            self._start_time = data.time[0]
            if dt is None and len(data.time) > 1:
                if not data.is_equidistant:
                    datetimes = data.time
                self._dt = (data.time[1] - data.time[0]).total_seconds()
            self._data = data.data
        else:
            self._data = data

        if start_time is None:
            if self._start_time is None:
                self._start_time = datetime.now()
                warnings.warn(
                    f"No start time supplied. Using current time: {start_time} as start time."
                )
            else:
                self._start_time = self._start_time
                warnings.warn(
                    f"No start time supplied. Using start time from source: {start_time} as start time."
                )
        else:
            self._start_time = start_time

        if items:
            self._items = items

        if self._items is None:
            self._items = [ItemInfo(f"Item {i+1}") for i in range(self._n_items)]

    def _setup_header(
        self, coordinate, start_time, dt, timeseries_unit, items, filename
    ):

        system_start_time = to_dotnet_datetime(self._start_time)

        self._builder.SetDataType(0)

        if self._coordinate[0] == "LONG/LAT":
            self._builder.SetGeographicalProjection(
                self._factory.CreateProjectionGeoOrigin(*self._coordinate)
            )
        else:
            self._builder.SetGeographicalProjection(
                self._factory.CreateProjectionProjOrigin(*self._coordinate)
            )

        if self._is_equidistant:
            self._builder.SetTemporalAxis(
                self._factory.CreateTemporalEqCalendarAxis(
                    timeseries_unit, system_start_time, 0, dt
                )
            )
        else:
            self._builder.SetTemporalAxis(
                self._factory.CreateTemporalNonEqCalendarAxis(
                    timeseries_unit, system_start_time
                )
            )

        for item in self._items:
            self._builder.AddDynamicItem(
                item.name,
                eumQuantity.Create(item.type, item.unit),
                DfsSimpleType.Float,
                DataValueType.Instantaneous,
            )

        try:
            self._builder.CreateFile(filename)
        except IOError:
            # TODO does this make sense?
            print("cannot create dfs file: ", filename)

        return self._builder.GetFile()

