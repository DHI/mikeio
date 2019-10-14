import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import clr
import System
from System import Array
from DHI.Generic.MikeZero import eumUnit, eumQuantity
from DHI.Generic.MikeZero.DFS import DfsFileFactory, DfsFactory, DfsSimpleType, DataValueType, DfsBuilder
from DHI.Generic.MikeZero.DFS.dfs123 import Dfs1Builder

from .dutil import to_numpy
from .helpers import safe_length


def dfs2todfs1(dfs2file, dfs1file, ax=0, func=np.nanmean):
    """ Function: take average (or other statistics) over axis in dfs2 and output to dfs1

    Usage:
        dfs2todfs1(dfs2file, dfs1file)
        dfs2todfs1(dfs2file, dfs1file, axis)
        dfs2todfs1(dfs2file, dfs1file, axis, func=np.nanmean)
    """

    # Read dfs2
    dfs_in = DfsFileFactory.DfsGenericOpen(dfs2file)
    fileInfo = dfs_in.FileInfo

    # Basic info from input file
    axis = dfs_in.ItemInfo[0].SpatialAxis
    n_time_steps = fileInfo.TimeAxis.NumberOfTimeSteps
    if n_time_steps == 0:
        raise Warning("Static dfs2 files (with no time steps) are not supported.")

    # Create an empty dfs1 file object
    factory = DfsFactory()
    builder = Dfs1Builder.Create(fileInfo.FileTitle, fileInfo.ApplicationTitle, fileInfo.ApplicationVersion)

    # Set up the header
    builder.SetDataType(fileInfo.DataType)
    builder.SetGeographicalProjection(fileInfo.Projection)
    builder.SetTemporalAxis(fileInfo.TimeAxis)
    builder.DeleteValueByte = fileInfo.DeleteValueByte
    builder.DeleteValueDouble = fileInfo.DeleteValueDouble
    builder.DeleteValueFloat = fileInfo.DeleteValueFloat
    builder.DeleteValueInt = fileInfo.DeleteValueInt
    builder.DeleteValueUnsignedInt = fileInfo.DeleteValueUnsignedInt

    # use x-axis (default) else y-axis
    if ax == 0:
        builder.SetSpatialAxis(factory.CreateAxisEqD1(axis.AxisUnit, axis.XCount, axis.X0, axis.Dx))
    else:
        builder.SetSpatialAxis(factory.CreateAxisEqD1(axis.AxisUnit, axis.YCount, axis.Y0, axis.Dy))

    # assume no compression keys
    if fileInfo.IsFileCompressed:
        raise Warning("Compressed files not supported")

    # custom blocks
    #cb = fileInfo.CustomBlocks #[0]
    #for j in range(safe_length(cb)):
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
        builder.AddDynamicItem(ii.Name, ii.Quantity, DfsSimpleType.Float, DataValueType.Instantaneous)

    try:
        builder.CreateFile(dfs1file)
    except IOError:
        print('cannot create dfs1 file: ', dfs1file)

    dfs_out = builder.GetFile()

    # read-write data
    deleteValue = fileInfo.DeleteValueFloat
    for it in range(n_time_steps):
        for item in range(n_items):
            itemdata = dfs_in.ReadItemTimeStep(item+1, it)

            d = to_numpy(itemdata.Data)
            d[d == deleteValue] = np.nan
            d2 = d.reshape(axis.YCount, axis.XCount)
            d2 = np.flipud(d2)

            d1 = func(d2, axis=ax)
            d1[np.isnan(d1)] = deleteValue

            darray = Array[System.Single](d1)
            dfs_out.WriteItemTimeStepNext(itemdata.Time, darray)

    dfs_in.Close()
    dfs_out.Close()


def dfstodfs0(dfsfile, dfs0file, func=np.nanmean):
    """ Function: take average (or other statistics) over dfs and output dfs0

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
        raise Warning("Static dfs files (with no time steps) are not supported.")

    # Create an empty dfs1 file object
    factory = DfsFactory()
    builder = DfsBuilder.Create(fileInfo.FileTitle, fileInfo.ApplicationTitle, fileInfo.ApplicationVersion)

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
        #itemj.SetReferenceCoordinates(0, 0, 0)
        builder.AddDynamicItem(itemj.GetDynamicItemInfo())

    try:
        builder.CreateFile(dfs0file)
    except IOError:
        print('cannot create dfs0 file: ', dfs0file)

    dfs_out = builder.GetFile()

    # read-write data
    deleteValue = fileInfo.DeleteValueFloat
    for it in range(n_time_steps):
        for item in range(n_items):
            itemdata = dfs_in.ReadItemTimeStep(item+1, it)

            d = to_numpy(itemdata.Data)
            d[d == deleteValue] = np.nan

            d0   = func(d)
            d    = np.zeros(1)
            d[0] = d0
            d[np.isnan(d)] = deleteValue

            darray = Array[System.Single](d)
            dfs_out.WriteItemTimeStepNext(itemdata.Time, darray)

    dfs_in.Close()
    dfs_out.Close()
