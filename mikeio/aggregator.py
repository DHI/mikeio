import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from mikecore.eum import eumQuantity, eumUnit
from mikecore.DfsFactory import DfsFactory, DfsBuilder, DfsSimpleType, DataValueType
from mikecore.DfsFileFactory import DfsFileFactory

from .helpers import safe_length


def dfs2todfs1(dfs2file, dfs1file, axis=1, func=np.nanmean):
    """Aggregate file over an axis

    Parameters
    ----------
    dfs2file : str
        input file
    dfs1file : str
        output file
    axis : int, optional
        spatial axis to aggregate over, 1=y, 2=x default 1
    func : function, optional
        aggregation function, by default np.nanmean
    """

    if axis not in [1, 2]:
        raise ValueError("Axis must be 1=y or 2=x")

    # Read dfs2
    dfs_in = DfsFileFactory.DfsGenericOpen(dfs2file)
    fileInfo = dfs_in.FileInfo

    # Basic info from input file
    ax = dfs_in.ItemInfo[0].SpatialAxis
    n_time_steps = fileInfo.TimeAxis.NumberOfTimeSteps
    if n_time_steps == 0:
        raise ValueError("Static dfs2 files (with no time steps) are not supported.")

    # Create an empty dfs1 file object
    factory = DfsFactory()
    builder = DfsBuilder.Create(
        fileInfo.FileTitle, fileInfo.ApplicationTitle, fileInfo.ApplicationVersion
    )

    # Set up the header
    builder.SetDataType(fileInfo.DataType)
    builder.SetGeographicalProjection(fileInfo.Projection)
    builder.SetTemporalAxis(fileInfo.TimeAxis)
    builder.DeleteValueByte = fileInfo.DeleteValueByte
    builder.DeleteValueDouble = fileInfo.DeleteValueDouble
    builder.DeleteValueFloat = fileInfo.DeleteValueFloat
    builder.DeleteValueInt = fileInfo.DeleteValueInt
    builder.DeleteValueUnsignedInt = fileInfo.DeleteValueUnsignedInt

    if axis == 1:
        builder.SetSpatialAxis(
            factory.CreateAxisEqD1(ax.AxisUnit, ax.XCount, ax.X0, ax.Dx)
        )
    else:
        builder.SetSpatialAxis(
            factory.CreateAxisEqD1(ax.AxisUnit, ax.YCount, ax.Y0, ax.Dy)
        )

    # assume no compression keys
    if fileInfo.IsFileCompressed:
        raise ValueError("Compressed files not supported")

    # custom blocks
    # cb = fileInfo.CustomBlocks #[0]
    # for j in range(safe_length(cb)):
    #    builder.AddCustomBlocks(cb[j])

    # static items
    while True:
        static_item = dfs_in.ReadStaticItemNext()
        if static_item == None:
            break
        builder.AddStaticItem(static_item)

    # dynamic items
    n_items = safe_length(dfs_in.ItemInfo)
    for item in range(n_items):
        ii = dfs_in.ItemInfo[item]
        builder.AddCreateDynamicItem(
            ii.Name, ii.Quantity, DfsSimpleType.Float, DataValueType.Instantaneous
        )

    try:
        builder.CreateFile(dfs1file)
    except IOError:
        print("cannot create dfs1 file: ", dfs1file)

    dfs_out = builder.GetFile()

    # read-write data
    deleteValue = fileInfo.DeleteValueFloat
    for it in range(n_time_steps):
        for item in range(n_items):
            itemdata = dfs_in.ReadItemTimeStep(item + 1, it)

            # d = to_numpy(itemdata.Data)
            d = itemdata.Data
            d[d == deleteValue] = np.nan
            d2 = d.reshape(ax.YCount, ax.XCount)
            d2 = np.flipud(d2)

            d1 = func(d2, axis=axis - 1)
            d1[np.isnan(d1)] = deleteValue

            # darray = to_dotnet_float_array(d1)
            darray = d1
            dfs_out.WriteItemTimeStepNext(itemdata.Time, darray)

    dfs_in.Close()
    dfs_out.Close()


def dfstodfs0(dfsfile, dfs0file, func=np.nanmean):
    """Function: take average (or other statistics) over dfs and output dfs0

    Usage:
        dfstodfs0(dfsfile, dfs0file)
        dfstodfs0(dfsfile, dfs0file, func=np.nanmean)
    """

    # Read dfs
    dfs_in = DfsFileFactory.DfsGenericOpen(dfsfile)
    fileInfo = dfs_in.FileInfo

    # Basic info from input file
    n_time_steps = fileInfo.TimeAxis.NumberOfTimeSteps
    if n_time_steps == 0:
        raise ValueError("Static dfs files (with no time steps) are not supported.")

    # Create an empty dfs1 file object
    factory = DfsFactory()
    builder = DfsBuilder.Create(
        fileInfo.FileTitle, fileInfo.ApplicationTitle, fileInfo.ApplicationVersion
    )

    # Set up the header
    builder.SetDataType(fileInfo.DataType)
    builder.SetGeographicalProjection(factory.CreateProjectionUndefined())
    builder.SetTemporalAxis(fileInfo.TimeAxis)
    builder.DeleteValueByte = fileInfo.DeleteValueByte
    builder.DeleteValueDouble = fileInfo.DeleteValueDouble
    builder.DeleteValueFloat = fileInfo.DeleteValueFloat
    builder.DeleteValueInt = fileInfo.DeleteValueInt
    builder.DeleteValueUnsignedInt = fileInfo.DeleteValueUnsignedInt

    # dynamic items
    n_items = safe_length(dfs_in.ItemInfo)
    for item in range(n_items):
        ii = dfs_in.ItemInfo[item]
        itemj = builder.CreateDynamicItemBuilder()
        itemj.Set(ii.Name, ii.Quantity, DfsSimpleType.Float)
        itemj.SetValueType(DataValueType.Instantaneous)
        itemj.SetAxis(factory.CreateAxisEqD0())
        # itemj.SetReferenceCoordinates(0, 0, 0)
        builder.AddDynamicItem(itemj.GetDynamicItemInfo())

    try:
        builder.CreateFile(dfs0file)
    except IOError:
        print("cannot create dfs0 file: ", dfs0file)

    dfs_out = builder.GetFile()

    # read-write data
    deleteValue = fileInfo.DeleteValueFloat
    for it in range(n_time_steps):
        for item in range(n_items):
            itemdata = dfs_in.ReadItemTimeStep(item + 1, it)

            # d = to_numpy(itemdata.Data)
            d = itemdata.Data
            d[d == deleteValue] = np.nan

            d0 = func(d)
            d = np.zeros(1)
            d[0] = d0
            d[np.isnan(d)] = deleteValue

            # darray = to_dotnet_float_array(d)
            darray = d.astype(np.float32)
            dfs_out.WriteItemTimeStepNext(itemdata.Time, darray)

    dfs_in.Close()
    dfs_out.Close()
