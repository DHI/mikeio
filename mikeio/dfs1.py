import numpy as np
from datetime import timedelta

from DHI.Generic.MikeZero import eumUnit
from DHI.Generic.MikeZero.DFS import (
    DfsFileFactory,
    DfsFactory,
    DfsSimpleType,
    DataValueType,
)
from DHI.Generic.MikeZero.DFS.dfs123 import Dfs1Builder

from .dutil import Dataset, find_item, get_item_info, get_valid_items_and_timesteps
from .dotnet import (
    to_numpy,
    to_dotnet_float_array,
    to_dotnet_datetime,
    from_dotnet_datetime,
)
from .eum import TimeStep, ItemInfo
from .helpers import safe_length
from .dfs import Dfs123


class Dfs1(Dfs123):

    _dx = None

    def __init__(self, filename=None):
        super(Dfs1, self).__init__(filename)

        if filename:
            self._read_dfs1_header()

    def _read_dfs1_header(self):
        dfs = DfsFileFactory.Dfs1FileOpen(self._filename)
        self._dx = dfs.SpatialAxis.Dx

        self._read_header(dfs)

    def read(self, items=None, time_steps=None):
        """
        Read data from a dfs1 file
        
        Parameters
        ---------
        items: list[int] or list[str], optional
            Read only selected items, by number (0-based), or by name
        time_steps: int or list[int], optional
            Read only selected time_steps

        Returns
        -------
        Dataset
            A dataset with data dimensions [t,x]
        """

        # NOTE. Item numbers are base 0 (everything else in the dfs is base 0)

        # Open the dfs file for reading
        dfs = DfsFileFactory.DfsGenericOpen(self._filename)
        self._dfs = dfs
        self._source = dfs

        nt = dfs.FileInfo.TimeAxis.NumberOfTimeSteps

        items, item_numbers, time_steps = get_valid_items_and_timesteps(
            self, items, time_steps
        )

        # Determine the size of the grid
        axis = dfs.ItemInfo[0].SpatialAxis

        xNum = axis.XCount
        nt = dfs.FileInfo.TimeAxis.NumberOfTimeSteps
        if nt == 0:
            raise ValueError(
                "Static dfs1 files (with no time steps) are not supported."
            )

        deleteValue = dfs.FileInfo.DeleteValueFloat

        n_items = len(item_numbers)
        data_list = []

        for item in range(n_items):
            # Initialize an empty data block
            data = np.ndarray(shape=(len(time_steps), xNum), dtype=float)
            data_list.append(data)

        t_seconds = np.zeros(len(time_steps), dtype=float)

        for i in range(len(time_steps)):
            it = time_steps[i]
            for item in range(n_items):

                itemdata = dfs.ReadItemTimeStep(item_numbers[item] + 1, it)

                src = itemdata.Data
                d = to_numpy(src)

                d[d == deleteValue] = np.nan
                data_list[item][it, :] = d

            t_seconds[it] = itemdata.Time

        start_time = from_dotnet_datetime(dfs.FileInfo.TimeAxis.StartDateTime)
        time = [start_time + timedelta(seconds=tsec) for tsec in t_seconds]

        items = get_item_info(dfs, item_numbers)

        dfs.Close()
        return Dataset(data_list, time, items)

    def write(
        self,
        filename,
        data,
        start_time=None,
        dt=1,
        items=None,
        dx=1,
        x0=0,
        coordinate=None,
        title=None,
    ):
        """
        Write a dfs1 file

        Parameters
        ----------
        filename: str
            Location to write the dfs1 file
        data: list[np.array]
            list of matrices, one for each item. Matrix dimension: x, time
        start_time: datetime, optional
            start datetime
        dt: float
            The time step in seconds. 
        items: list[ItemInfo], optional
            List of ItemInfo corresponding to a variable types (ie. Water Level).
        coordinate:
            ['UTM-33', 12.4387, 55.2257, 327]  for UTM, Long, Lat, North to Y orientation. Note: long, lat in decimal degrees
        x0:
            Lower right position
        dx:
            length of each grid in the x direction (meters)
        title: str, optional
            title of the dfs file (can be blank)

        """

        self._write_handle_common_arguments(
            title, data, items, coordinate, start_time, dt
        )

        number_x = np.shape(data[0])[1]

        if dx is None:
            if self._dx is not None:
                dx = self._dx
            else:
                dx = 1

        if not all(np.shape(d)[1] == number_x for d in data):
            raise ValueError(
                "ERROR data matrices in the X dimension do not all match in the data list. "
                "Data is list of matices [t, x]"
            )

        factory = DfsFactory()
        builder = Dfs1Builder.Create(title, "mikeio", 0)

        self._builder = builder
        self._factory = factory

        builder.SetSpatialAxis(
            factory.CreateAxisEqD1(eumUnit.eumUmeter, number_x, x0, dx)
        )

        dfs = self._setup_header(filename)

        deletevalue = dfs.FileInfo.DeleteValueFloat  # -1.0000000031710769e-30

        for i in range(self._n_time_steps):
            for item in range(self._n_items):
                d = data[item][i, :]
                d[np.isnan(d)] = deletevalue

                darray = to_dotnet_float_array(d)
                dfs.WriteItemTimeStepNext(0, darray)

        dfs.Close()
