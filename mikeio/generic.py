import os
from datetime import datetime, timedelta

from DHI.Generic.MikeZero.DFS import DfsFileFactory, DfsBuilder
from .dotnet import to_numpy, to_dotnet_float_array, from_dotnet_datetime
from .helpers import safe_length
from shutil import copyfile


def _clone(infilename, outfilename):
    """Clone a dfs file

    Parameters
    ----------
    infilename : str
        input filename
    outfilename : str
        output filename

    Returns
    -------
    DfsFile
        MIKE generic dfs file object
    """
    source = DfsFileFactory.DfsGenericOpen(infilename)
    fi = source.FileInfo

    builder = DfsBuilder.Create(
        fi.FileTitle, fi.ApplicationTitle, fi.ApplicationVersion
    )

    # Set up the header
    builder.SetDataType(fi.DataType)
    builder.SetGeographicalProjection(fi.Projection)
    builder.SetTemporalAxis(fi.TimeAxis)
    builder.SetItemStatisticsType(fi.StatsType)
    builder.DeleteValueByte = fi.DeleteValueByte
    builder.DeleteValueDouble = fi.DeleteValueDouble
    builder.DeleteValueFloat = fi.DeleteValueFloat
    builder.DeleteValueInt = fi.DeleteValueInt
    builder.DeleteValueUnsignedInt = fi.DeleteValueUnsignedInt

    # Copy custom blocks - if any
    for customBlock in fi.CustomBlocks:
        builder.AddCustomBlock(customBlock)

    # Copy dynamic items
    for itemInfo in source.ItemInfo:
        builder.AddDynamicItem(itemInfo)

    # Create file
    builder.CreateFile(outfilename)

    # Copy static items
    while True:
        static_item = source.ReadStaticItemNext()
        if static_item is None:
            break
        builder.AddStaticItem(static_item)

    # Get the file
    file = builder.GetFile()

    return file


def scale(infilename, outfilename, offset=0.0, factor=1.0):
    """Apply scaling to any dfs file

        Parameters
        ----------

        infilename: str
            full path to the input file
        outfilename: str
            full path to the output file
        offset: float, optional
            value to add to all items, default 0.0
        factor: float, optional
            value to multiply to all items, default 1.0
        """

    copyfile(infilename, outfilename)

    dfs = DfsFileFactory.DfsGenericOpenEdit(outfilename)

    n_time_steps = dfs.FileInfo.TimeAxis.NumberOfTimeSteps
    n_items = safe_length(dfs.ItemInfo)

    for timestep in range(n_time_steps):
        for item in range(n_items):

            itemdata = dfs.ReadItemTimeStep(item + 1, timestep)
            d = to_numpy(itemdata.Data)
            time = itemdata.Time

            outdata = d * factor + offset
            darray = to_dotnet_float_array(outdata)

            dfs.WriteItemTimeStep(item + 1, timestep, time, darray)


def sum(infilename_a, infilename_b, outfilename):
    """Sum two dfs files (a+b)

    Parameters
    ----------
    infilename_a: str
        full path to the first input file
    infilename_b: str
        full path to the second input file
    outfilename: str
        full path to the output file
    """
    copyfile(infilename_a, outfilename)

    dfs_i_a = DfsFileFactory.DfsGenericOpen(infilename_a)
    dfs_i_b = DfsFileFactory.DfsGenericOpen(infilename_b)
    dfs_o = DfsFileFactory.DfsGenericOpenEdit(outfilename)

    n_time_steps = dfs_i_a.FileInfo.TimeAxis.NumberOfTimeSteps
    n_items = safe_length(dfs_i_a.ItemInfo)
    # TODO Add checks to verify identical structure of file a and b

    for timestep in range(n_time_steps):
        for item in range(n_items):

            itemdata_a = dfs_i_a.ReadItemTimeStep(item + 1, timestep)
            d_a = to_numpy(itemdata_a.Data)

            itemdata_b = dfs_i_b.ReadItemTimeStep(item + 1, timestep)
            d_b = to_numpy(itemdata_b.Data)
            time = itemdata_a.Time

            outdata = d_a + d_b

            darray = to_dotnet_float_array(outdata)

            dfs_o.WriteItemTimeStep(item + 1, timestep, time, darray)

    dfs_i_a.Close()
    dfs_i_b.Close()
    dfs_o.Close()


def diff(infilename_a, infilename_b, outfilename):
    """Calculate difference between two dfs files (a-b)

    Parameters
    ----------
    infilename_a : str
        full path to the first input file
    infilename_b : str
        full path to the second input file
    outfilename : str
        full path to the output file
    """

    copyfile(infilename_a, outfilename)

    dfs_i_a = DfsFileFactory.DfsGenericOpen(infilename_a)
    dfs_i_b = DfsFileFactory.DfsGenericOpen(infilename_b)
    dfs_o = DfsFileFactory.DfsGenericOpenEdit(outfilename)

    n_time_steps = dfs_i_a.FileInfo.TimeAxis.NumberOfTimeSteps
    n_items = safe_length(dfs_i_a.ItemInfo)
    # TODO Add checks to verify identical structure of file a and b

    for timestep in range(n_time_steps):
        for item in range(n_items):

            itemdata_a = dfs_i_a.ReadItemTimeStep(item + 1, timestep)
            d_a = to_numpy(itemdata_a.Data)

            itemdata_b = dfs_i_b.ReadItemTimeStep(item + 1, timestep)
            d_b = to_numpy(itemdata_b.Data)
            time = itemdata_a.Time

            outdata = d_a - d_b

            darray = to_dotnet_float_array(outdata)

            dfs_o.WriteItemTimeStep(item + 1, timestep, time, darray)

    dfs_i_a.Close()
    dfs_i_b.Close()
    dfs_o.Close()


def concat(infilenames, outfilename):
    """Concatenates files along the time axis

    If files are overlapping, the last one will be used.

    Parameters
    ----------
    infilenames: list of str
        filenames to concatenate

    outfilename: str
        filename

    Notes
    ------

    The list of input files have to be sorted, i.e. in chronological order
    """

    dfs_i_a = DfsFileFactory.DfsGenericOpen(infilenames[0])

    dfs_o = _clone(infilenames[0], outfilename)

    n_items = safe_length(dfs_i_a.ItemInfo)

    current_time = datetime(1,1,1) # beginning of time...

    for i, infilename in enumerate(infilenames):

        dfs_i = DfsFileFactory.DfsGenericOpen(infilename)
        t_axis = dfs_i.FileInfo.TimeAxis
        n_time_steps = t_axis.NumberOfTimeSteps
        dt = t_axis.TimeStep
        start_time = from_dotnet_datetime(t_axis.StartDateTime)

        if i > 0 and start_time > current_time + timedelta(seconds=dt):
            dfs_o.Close()
            os.remove(outfilename)
            raise Exception("Gap in time axis detected - not supported")

        current_time = start_time

        if i < (len(infilenames) - 1):
            dfs_n = DfsFileFactory.DfsGenericOpen(infilenames[i + 1])
            nf = dfs_n.FileInfo.TimeAxis.StartDateTime
            next_start_time = datetime(
                nf.Year, nf.Month, nf.Day, nf.Hour, nf.Minute, nf.Second
            )

        for timestep in range(n_time_steps):

            current_time = start_time + timedelta(seconds=timestep * dt)
            if i < (len(infilenames) - 1):
                if current_time >= next_start_time:
                    break

            for item in range(n_items):

                itemdata = dfs_i.ReadItemTimeStep(item + 1, timestep)
                d = to_numpy(itemdata.Data)

                darray = to_dotnet_float_array(d)

                dfs_o.WriteItemTimeStepNext(0, darray)

    dfs_o.Close()
