from .helpers import safe_length
from .dutil import Dataset, get_item_info, get_valid_items_and_timesteps
from .dotnet import (
    to_numpy,
    to_dotnet_float_array,
    to_dotnet_datetime,
    from_dotnet_datetime,
)

class Dfs123:

    _projstr = None

    def __init__(self, filename = None):
        self._filename = filename
        # TODO self._read_header(filename)

    def _read_header(self, dfs):
        self._n_items = safe_length(dfs.ItemInfo)
        self._items = get_item_info(dfs, list(range(self._n_items)))
        self._start_time = from_dotnet_datetime(dfs.FileInfo.TimeAxis.StartDateTime)
        self._n_timesteps = dfs.FileInfo.TimeAxis.NumberOfTimeSteps
        #self._timestep_in_seconds = dfs.FileInfo.TimeAxis.TimeStepInSeconds
        self._projstr = dfs.FileInfo.Projection.WKTString
        self._longitude = dfs.FileInfo.Projection.Longitude
        self._latitude = dfs.FileInfo.Projection.Latitude
        self._orientation = dfs.FileInfo.Projection.Orientation

        dfs.Close()
