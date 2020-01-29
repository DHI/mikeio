import numpy as np
from System.Runtime.InteropServices import GCHandle, GCHandleType
import ctypes

from collections import namedtuple


def to_numpy(src):

    src_hndl = GCHandle.Alloc(src, GCHandleType.Pinned)
    try:
        src_ptr = src_hndl.AddrOfPinnedObject().ToInt64()
        bufType = ctypes.c_float * len(src)
        cbuf = bufType.from_address(src_ptr)
        d = np.frombuffer(cbuf, dtype=cbuf._type_)
    finally:
        if src_hndl.IsAllocated: src_hndl.Free()

    return d


#Dataset = namedtuple("Dataset", ["data", "time", "names"])
class Dataset(namedtuple("Dataset", ["data", "time", "names"])):

    def __repr__(self):
        n_items = len(self.names)

        out = []
        out.append("DataSet(data, time, names)")
        out.append(f"Number of items: {n_items}")
        out.append(f"Shape: {self.data[0].shape}")
        out.append(f"{self.time[0]} - {self.time[-1]}")

        return str.join("\n", out)
