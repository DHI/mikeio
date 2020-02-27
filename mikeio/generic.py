import System
from System import Array
from DHI.Generic.MikeZero.DFS import DfsFileFactory
from .dutil import to_numpy
from .helpers import safe_length
from shutil import copyfile


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
