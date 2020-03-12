from datetime import datetime, timedelta
import System
from System import Array
from DHI.Generic.MikeZero.DFS import DfsFileFactory, DfsBuilder
from .dutil import to_numpy
from .helpers import safe_length
from shutil import copyfile


def _clone(infilename, outfilename):
    source = DfsFileFactory.DfsGenericOpen(infilename)
    fileInfo = source.FileInfo

    builder = DfsBuilder.Create(
        fileInfo.FileTitle, fileInfo.ApplicationTitle, fileInfo.ApplicationVersion
    )

    # Set up the header
    builder.SetDataType(fileInfo.DataType)
    builder.SetGeographicalProjection(fileInfo.Projection)
    builder.SetTemporalAxis(fileInfo.TimeAxis)
    builder.SetItemStatisticsType(fileInfo.StatsType)
    builder.DeleteValueByte = fileInfo.DeleteValueByte
    builder.DeleteValueDouble = fileInfo.DeleteValueDouble
    builder.DeleteValueFloat = fileInfo.DeleteValueFloat
    builder.DeleteValueInt = fileInfo.DeleteValueInt
    builder.DeleteValueUnsignedInt = fileInfo.DeleteValueUnsignedInt

    # Copy custom blocks - if any
    for customBlock in fileInfo.CustomBlocks:
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

        Usage:
            scale(infilename, outfilename, offset=0.0, factor=1.0):
        infilename
            full path to the input file
        outfilename
            full path to the output file
        offset
            value to add to all items, default 0.0
        factor
            value to multiply to all items, default 1.0

        Return:
            Nothing
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
            darray = Array[System.Single](outdata)

            dfs.WriteItemTimeStep(item + 1, timestep, time, darray)


def sum(infilename_a, infilename_b, outfilename):
    """Sum two dfs files

    Usage:
        sum(infilename_a, infilename_b, outfilename):
    infilename_a
        full path to the first input file
    infilename_b
        full path to the second input file
    outfilename
        full path to the output file
    
    Return:
        Nothing
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
            darray = Array[System.Single](outdata)

            dfs_o.WriteItemTimeStep(item + 1, timestep, time, darray)

    dfs_i_a.Close()
    dfs_i_b.Close()
    dfs_o.Close()


def diff(infilename_a, infilename_b, outfilename):
    """Caluclate difference between two dfs files

    Usage:
        diff(infilename_a, infilename_b, outfilename):
    infilename_a
        full path to the first input file
    infilename_b
        full path to the second input file
    outfilename
        full path to the output file
    
    Return:
        Nothing
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
            darray = Array[System.Single](outdata)

            dfs_o.WriteItemTimeStep(item + 1, timestep, time, darray)

    dfs_i_a.Close()
    dfs_i_b.Close()
    dfs_o.Close()


def concat(infilenames, outfilename):

    dfs_i_a = DfsFileFactory.DfsGenericOpen(infilenames[0])

    dfs_o = _clone(infilenames[0], outfilename)

    n_items = safe_length(dfs_i_a.ItemInfo)

    for i, infilename in enumerate(infilenames):

        dfs_i = DfsFileFactory.DfsGenericOpen(infilename)
        n_time_steps = dfs_i.FileInfo.TimeAxis.NumberOfTimeSteps
        dt = dfs_i.FileInfo.TimeAxis.TimeStep
        s = dfs_i.FileInfo.TimeAxis.StartDateTime
        start_time = datetime(s.Year, s.Month, s.Day, s.Hour, s.Minute, s.Second)

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

                darray = Array[System.Single](d)

                dfs_o.WriteItemTimeStepNext(0, darray)

    dfs_o.Close()
