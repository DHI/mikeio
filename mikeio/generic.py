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
