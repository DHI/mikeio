import numpy as np
from datetime import datetime, timedelta

from DHI.Generic.MikeZero import eumUnit, eumQuantity
from DHI.Generic.MikeZero.DFS import (
    DfsFileFactory,
    DfsFactory,
    DfsSimpleType,
    DataValueType,
)
from DHI.Generic.MikeZero.DFS.dfs123 import Dfs1Builder

from .dutil import Dataset, find_item, get_item_info
from .dotnet import to_numpy, to_dotnet_float_array, to_dotnet_datetime, from_dotnet_datetime
from .eum import TimeStep, ItemInfo
from .helpers import safe_length


class Dfs1:
    def read(self, filename, item_numbers=None, item_names=None):
        """Read data from the dfs1 file

        Usage:
            read(filename, item_numbers=None, item_names=None)
        filename
            full path to the dfs1 file.
        item_numbers
            read only the item_numbers in the array specified (0 base)
        item_names
            read only the items in the array specified, (takes precedence over item_numbers)

        Return:
            Dataset(data, time, items)
            where data[nt,x]
        """

        # NOTE. Item numbers are base 0 (everything else in the dfs is base 0)

        # Open the dfs file for reading
        dfs = DfsFileFactory.DfsGenericOpen(filename)

        if item_names is not None:
            item_numbers = find_item(dfs, item_names)

        if item_numbers is None:
            n_items = safe_length(dfs.ItemInfo)
            item_numbers = list(range(n_items))

        # Determine the size of the grid
        axis = dfs.ItemInfo[0].SpatialAxis

        xNum = axis.XCount
        nt = dfs.FileInfo.TimeAxis.NumberOfTimeSteps
        if nt == 0:
            raise Warning("Static dfs1 files (with no time steps) are not supported.")
            nt = 1
        deleteValue = dfs.FileInfo.DeleteValueFloat

        n_items = len(item_numbers)
        data_list = []

        for item in range(n_items):
            # Initialize an empty data block
            data = np.ndarray(shape=(nt, xNum), dtype=float)
            data_list.append(data)

        t_seconds = np.zeros(nt, dtype=float)
        
        for it in range(dfs.FileInfo.TimeAxis.NumberOfTimeSteps):
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

    def write(self, filename, data):
        """
        Function: write to a pre-created dfs1 file.

        filename:
            full path and filename to existing dfs1 file

        data:
            list of matrices. len(data) must equal the number of items in the dfs2.
            Each matrix must be of dimension time, x

        usage:
            write(filename, data) where  data(nt, x)

        Returns:
            Nothing

        """

        # Open the dfs file for writing
        dfs = DfsFileFactory.Dfs1FileOpenEdit(filename)

        # Determine the size of the grid
        number_x = dfs.SpatialAxis.XCount
        n_time_steps = dfs.FileInfo.TimeAxis.NumberOfTimeSteps
        n_items = safe_length(dfs.ItemInfo)

        deletevalue = -1e-035

        if not all(np.shape(d)[0] == n_time_steps for d in data):
            raise Warning(
                "ERROR data matrices in the time dimension do not all match in the data list. "
                "Data is list of matices [t, x]"
            )
        if not all(np.shape(d)[1] == number_x for d in data):
            raise Warning(
                "ERROR data matrices in the X dimension do not all match in the data list. "
                "Data is list of matices [t, x]"
            )

        if not len(data) == n_items:
            raise Warning(
                "The number of matrices in data do not match the number of items in the dfs1 file."
            )

        for i in range(n_time_steps):
            for item in range(n_items):
                d = data[item][i, :]
                d[np.isnan(d)] = deletevalue
                darray = to_dotnet_float_array(d)
                dfs.WriteItemTimeStepNext(0, darray)

        dfs.Close()

    def create(
        self,
        filename,
        data,
        start_time=None,
        dt=1,
        items=None,
        length_x=1,
        x0=0,
        coordinate=None,
        timeseries_unit=TimeStep.SECOND,
        title=None,
    ):
        """
        Create a dfs1 file

        Parameters
        ----------
        filename: str
            Location to write the dfs1 file
        data: list[np.array]
            list of matrices, one for each item. Matrix dimension: x, time
        start_time: datetime, optional
            start datetime
        timeseries_unit: Timestep, optional
            TimeStep unit default TimeStep.SECOND
        dt: float
            The time step (double based on the timeseries_unit). Therefore dt of 5.5 with timeseries_unit of minutes
            means 5 mins and 30 seconds.
        items: list[ItemInfo], optional
            List of ItemInfo corresponding to a variable types (ie. Water Level).
        coordinate:
            ['UTM-33', 12.4387, 55.2257, 327]  for UTM, Long, Lat, North to Y orientation. Note: long, lat in decimal degrees
            OR
            [TODO: Support not Local Coordinates ...]
        x0:
            Lower right position
        length_x:
            length of each grid in the x direction (meters)
        title:
            title of the dfs2 file (can be blank)

        """

        if title is None:
            title = ""

        n_time_steps = np.shape(data[0])[0]
        number_x = np.shape(data[0])[1]
        n_items = len(data)

        if start_time is None:
            start_time = datetime.now()

        if coordinate is None:
            coordinate = ["LONG/LAT", 0, 0, 0]

        if items is None:
            items = [ItemInfo(f"temItem {i+1}") for i in range(n_items)]

        if not all(np.shape(d)[0] == n_time_steps for d in data):
            raise Warning(
                "ERROR data matrices in the time dimension do not all match in the data list. "
                "Data is list of matices [t, x]"
            )
        if not all(np.shape(d)[1] == number_x for d in data):
            raise Warning(
                "ERROR data matrices in the X dimension do not all match in the data list. "
                "Data is list of matices [t, x]"
            )

        if len(items) != n_items:
            raise Warning(
                "names must be an array of strings with the same number as matrices in data list"
            )

        if not type(start_time) is datetime:
            raise Warning("start_time must be of type datetime ")

        system_start_time = to_dotnet_datetime(start_time)

        # Create an empty dfs1 file object
        factory = DfsFactory()
        builder = Dfs1Builder.Create(title, "mikeio", 0)

        # Set up the header
        builder.SetDataType(0)
        builder.SetGeographicalProjection(
            factory.CreateProjectionGeoOrigin(
                coordinate[0], coordinate[1], coordinate[2], coordinate[3]
            )
        )
        builder.SetTemporalAxis(
            factory.CreateTemporalEqCalendarAxis(
                timeseries_unit, system_start_time, 0, dt
            )
        )
        builder.SetSpatialAxis(
            factory.CreateAxisEqD1(eumUnit.eumUmeter, number_x, x0, length_x)
        )

        for i in range(n_items):
            builder.AddDynamicItem(
                items[i].name,
                eumQuantity.Create(items[i].type, items[i].unit),
                DfsSimpleType.Float,
                DataValueType.Instantaneous,
            )

        try:
            builder.CreateFile(filename)
        except IOError:
            print("cannot create dfs2 file: ", filename)

        dfs = builder.GetFile()
        deletevalue = dfs.FileInfo.DeleteValueFloat  # -1.0000000031710769e-30

        for i in range(n_time_steps):
            for item in range(n_items):
                d = data[item][i, :]
                d[np.isnan(d)] = deletevalue
                # darray = Array[System.Single](np.array(d.reshape(d.size, 1)[:, 0]))
                darray = to_dotnet_float_array(d)
                dfs.WriteItemTimeStepNext(0, darray)

        dfs.Close()
