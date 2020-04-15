import numpy as np
import ctypes
import clr
import System
from System.Runtime.InteropServices import GCHandle, GCHandleType

_MAP_NP_NET = {
    np.dtype('float32'): System.Single,
    np.dtype('float64'): System.Double,
    np.dtype('int8')   : System.SByte,
    np.dtype('int16')  : System.Int16,
    np.dtype('int32')  : System.Int32,
    np.dtype('int64')  : System.Int64,
    np.dtype('uint8')  : System.Byte,
    np.dtype('uint16') : System.UInt16,
    np.dtype('uint32') : System.UInt32,
    np.dtype('uint64') : System.UInt64,
    np.dtype('bool')   : System.Boolean,
}
_MAP_NET_NP = {
    'Single' : np.dtype('float32'),
    'Double' : np.dtype('float64'),
    'SByte'  : np.dtype('int8'),
    'Int16'  : np.dtype('int16'), 
    'Int32'  : np.dtype('int32'),
    'Int64'  : np.dtype('int64'),
    'Byte'   : np.dtype('uint8'),
    'UInt16' : np.dtype('uint16'),
    'UInt32' : np.dtype('uint32'),
    'UInt64' : np.dtype('uint64'),
    'Boolean': np.dtype('bool'),
}

def asNumpyArray(netArray):
    '''
    Given a CLR `System.Array` returns a `numpy.ndarray`.  See _MAP_NET_NP for 
    the mapping of CLR types to Numpy dtypes.
    '''
    dims = np.empty(netArray.Rank, dtype=int)
    for I in range(netArray.Rank):
        dims[I] = netArray.GetLength(I)
    netType = netArray.GetType().GetElementType().Name

    try:
        npArray = np.empty(dims, order='C', dtype=_MAP_NET_NP[netType])
    except KeyError:
        raise NotImplementedError("asNumpyArray does not yet support System type {}".format(netType) )

    try: # Memmove 
        sourceHandle = GCHandle.Alloc(netArray, GCHandleType.Pinned)
        sourcePtr = sourceHandle.AddrOfPinnedObject().ToInt64()
        destPtr = npArray.__array_interface__['data'][0]
        ctypes.memmove(destPtr, sourcePtr, npArray.nbytes)
    finally:
        if sourceHandle.IsAllocated: sourceHandle.Free()
    return npArray

def asNetArray(npArray):
    '''
    Given a `numpy.ndarray` returns a CLR `System.Array`.  See _MAP_NP_NET for 
    the mapping of Numpy dtypes to CLR types.

    Note: `complex64` and `complex128` arrays are converted to `float32` 
    and `float64` arrays respectively with shape [m,n,...] -> [m,n,...,2]
    '''
    dims = npArray.shape
    dtype = npArray.dtype
    # For complex arrays, we must make a view of the array as its corresponding 
    # float type.
    if dtype == np.complex64:
        dtype = np.dtype('float32')
        dims.append(2)
        npArray = npArray.view(np.float32).reshape(dims)
    elif dtype == np.complex128:
        dtype = np.dtype('float64')
        dims.append(2)
        npArray = npArray.view(np.float64).reshape(dims)

    netDims = System.Array.CreateInstance(System.Int32, npArray.ndim)
    for I in range(npArray.ndim):
        netDims[I] = System.Int32(dims[I])
    
    if not npArray.flags.c_contiguous:
        npArray = npArray.copy(order='C')
    assert npArray.flags.c_contiguous

    try:
        netArray = System.Array.CreateInstance(_MAP_NP_NET[dtype], netDims)
    except KeyError:
        raise NotImplementedError("asNetArray does not yet support dtype {}".format(dtype))

    try: # Memmove 
        destHandle = GCHandle.Alloc(netArray, GCHandleType.Pinned)
        sourcePtr = npArray.__array_interface__['data'][0]
        destPtr = destHandle.AddrOfPinnedObject().ToInt64()
        ctypes.memmove(destPtr, sourcePtr, npArray.nbytes)
    finally:
        if destHandle.IsAllocated: destHandle.Free()
    return netArray


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
