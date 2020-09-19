from datetime import datetime, timedelta
import warnings
import numpy as np
from .helpers import safe_length
from .dutil import Dataset, get_item_info
from .dotnet import (
    to_dotnet_datetime,
    from_dotnet_datetime,
)
from .eum import ItemInfo, TimeStepUnit
from DHI.Generic.MikeZero import eumQuantity
from DHI.Generic.MikeZero.DFS import (
    DfsSimpleType,
    DataValueType,
)


class Dfs123:

    _filename = None
    _projstr = None
    _start_time = None
    _is_equidistant = True
    _items = None
    _builder = None
    _factory = None
    _deletevalue = None
    _override_coordinates = False
    _timeseries_unit = TimeStepUnit.SECOND
    _dt = None

    def __init__(self, filename=None):
        self._filename = filename

    def _read_header(self, dfs):
        self._n_items = safe_length(dfs.ItemInfo)
        self._items = get_item_info(dfs, list(range(self._n_items)))
        self._start_time = from_dotnet_datetime(dfs.FileInfo.TimeAxis.StartDateTime)
        if hasattr(dfs.FileInfo.TimeAxis, "TimeStep"):
            self._timestep_in_seconds = (
                dfs.FileInfo.TimeAxis.TimeStep
            )  # TODO handle other timeunits
        self._n_timesteps = dfs.FileInfo.TimeAxis.NumberOfTimeSteps
        self._projstr = dfs.FileInfo.Projection.WKTString
        self._longitude = dfs.FileInfo.Projection.Longitude
        self._latitude = dfs.FileInfo.Projection.Latitude
        self._orientation = dfs.FileInfo.Projection.Orientation
        self._deletevalue = dfs.FileInfo.DeleteValueFloat

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
            self._override_coordinates = True
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
            warnings.warn(f"No timestep supplied. Using 1s.")

        if items:
            self._items = items

        if self._items is None:
            self._items = [ItemInfo(f"Item {i+1}") for i in range(self._n_items)]

        self._timeseries_unit = TimeStepUnit.SECOND

    def _setup_header(self, filename):

        system_start_time = to_dotnet_datetime(self._start_time)

        self._builder.SetDataType(0)

        if self._coordinate[0] == "LONG/LAT":
            proj = self._factory.CreateProjectionGeoOrigin(*self._coordinate)
        else:
            if self._override_coordinates:
                proj = self._factory.CreateProjectionProjOrigin(*self._coordinate)
            else:
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

    @property
    def deletevalue(self):
        """File delete value
        """
        return self._deletevalue

    @property
    def n_items(self):
        """Number of items
        """
        return self._n_items

    @property
    def items(self):
        """List of items
        """
        return self._items

    @property
    def start_time(self):
        """File start time
        """
        return self._start_time

    @property
    def n_timesteps(self):
        """Number of time steps
        """
        return self._n_timesteps

    @property
    def timestep(self):
        """Time step size in seconds
        """
        return self._timestep_in_seconds

    @property
    def projection_string(self):
        return self._projstr

    @property
    def longitude(self):
        """Origin longitude
        """
        return self._longitude

    @property
    def latitude(self):
        """Origin latitude
        """
        return self._latitude

    @property
    def orientation(self):
        """North to Y orientation
        """
        return self._orientation
