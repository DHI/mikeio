import datetime
import numpy as np
import ctypes
import clr
import System
from System.Runtime.InteropServices import GCHandle, GCHandleType

_MAP_NP_NET = {
    np.dtype("float32"): System.Single,
    np.dtype("float64"): System.Double,
    np.dtype("int8"): System.SByte,
    np.dtype("int16"): System.Int16,
    np.dtype("int32"): System.Int32,
    np.dtype("int64"): System.Int64,
    np.dtype("uint8"): System.Byte,
    np.dtype("uint16"): System.UInt16,
    np.dtype("uint32"): System.UInt32,
    np.dtype("uint64"): System.UInt64,
    np.dtype("bool"): System.Boolean,
}
_MAP_NET_NP = {
    "Single": np.dtype("float32"),
    "Double": np.dtype("float64"),
    "SByte": np.dtype("int8"),
    "Int16": np.dtype("int16"),
    "Int32": np.dtype("int32"),
    "Int64": np.dtype("int64"),
    "Byte": np.dtype("uint8"),
    "UInt16": np.dtype("uint16"),
    "UInt32": np.dtype("uint32"),
    "UInt64": np.dtype("uint64"),
    "Boolean": np.dtype("bool"),
}


def to_dotnet_datetime(x):
    """Convert from python datetime to .NET System.DateTime """
    return System.DateTime(x.year, x.month, x.day, x.hour, x.minute, x.second,)


def from_dotnet_datetime(x):
    """Convert from .NET System.DateTime to python datetime"""
    return datetime.datetime(x.Year, x.Month, x.Day, x.Hour, x.Minute, x.Second)


def asNumpyArray(x):
    """
    Convert .NET array to numpy array
    Parameters
    ----------
    x : System.Array
        
    Returns
    -------
    np.ndarray
        
    Notes
    -----
    Given a CLR `System.Array` returns a `numpy.ndarray`.  See _MAP_NET_NP for 
    the mapping of CLR types to Numpy dtypes.
    """
    dims = np.empty(x.Rank, dtype=int)
    for I in range(x.Rank):
        dims[I] = x.GetLength(I)
    netType = x.GetType().GetElementType().Name

    try:
        npArray = np.empty(dims, order="C", dtype=_MAP_NET_NP[netType])
    except KeyError:
        raise NotImplementedError(
            "asNumpyArray does not yet support System type {}".format(netType)
        )

    try:  # Memmove
        sourceHandle = GCHandle.Alloc(x, GCHandleType.Pinned)
        sourcePtr = sourceHandle.AddrOfPinnedObject().ToInt64()
        destPtr = npArray.__array_interface__["data"][0]
        ctypes.memmove(destPtr, sourcePtr, npArray.nbytes)
    finally:
        if sourceHandle.IsAllocated:
            sourceHandle.Free()
    return npArray


def to_dotnet_array(x):
    """
    Convert numpy array to .NET array with same data type (single, double,...)

    Parameters
    ----------
    x: np.array

    Returns
    -------
    System.Array
        
    Notes
    -----
    Given a `numpy.ndarray` returns a CLR `System.Array`.  See _MAP_NP_NET for 
    the mapping of Numpy dtypes to CLR types.
    """
    dims = x.shape
    dtype = x.dtype

    if not x.flags.c_contiguous:
        x = x.copy(order="C")
    assert x.flags.c_contiguous

    try:
        netArray = System.Array.CreateInstance(_MAP_NP_NET[dtype], dims)
    except KeyError:
        raise NotImplementedError(
            "asNetArray does not yet support dtype {}".format(dtype)
        )

    try:  # Memmove
        destHandle = GCHandle.Alloc(netArray, GCHandleType.Pinned)
        sourcePtr = x.__array_interface__["data"][0]
        destPtr = destHandle.AddrOfPinnedObject().ToInt64()
        ctypes.memmove(destPtr, sourcePtr, x.nbytes)
    finally:
        if destHandle.IsAllocated:
            destHandle.Free()
    return netArray


def asnetarray_v2(x):
    if any([type(xi) is list for xi in x]):
        # Array of array
        return asnetarray_v2([asnetarray_v2(xi) for xi in x])
    else:
        # Array
        return System.Array[type(x[1])](x)


def to_dotnet_float_array(x):

    return to_dotnet_array(x.astype(np.float32))


def to_numpy(src):
    """
    Convert .NET array to numpy array

    Parameters
    ----------
    src : System.Array
        
    Returns
    -------
    np.ndarray
        
    """

    src_hndl = GCHandle.Alloc(src, GCHandleType.Pinned)
    try:
        src_ptr = src_hndl.AddrOfPinnedObject().ToInt64()
        bufType = ctypes.c_float * len(src)
        cbuf = bufType.from_address(src_ptr)
        d = np.frombuffer(cbuf, dtype=cbuf._type_)
    finally:
        if src_hndl.IsAllocated:
            src_hndl.Free()

    return d
