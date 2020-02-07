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
        if src_hndl.IsAllocated:
            src_hndl.Free()

    return d


def find_item(dfs, item_names):
    names = [x.Name for x in dfs.ItemInfo]
    item_lookup = {name: i for i, name in enumerate(names)}
    try:
        item_numbers = [item_lookup[x] for x in item_names]
    except:
        raise ValueError(f"Selected item name not found. Valid names are {names}")
    return item_numbers


class Dataset:
    def __init__(self, data, time, names):
        self.data = data
        self.time = time
        self.names = names

    def __repr__(self):
        n_items = len(self.names)

        out = []
        out.append("DataSet(data, time, names)")
        out.append(f"Number of items: {n_items}")
        out.append(f"Shape: {self.data[0].shape}")
        out.append(f"{self.time[0]} - {self.time[-1]}")

        return str.join("\n", out)

    def __len__(self):
        return 2  # [data,time,names]

    def __getitem__(self, x):

        if isinstance(x, int):
            if x == 0:
                return self.data
            if x == 1:
                return self.time
            if x == 2:
                return self.names

            if x > 2:
                raise IndexError("")

        if isinstance(x, str):
            item_lookup = {name: i for i, name in enumerate(self.names)}
            x = item_lookup[x]
            return self.data[x]

        raise Exception("Invalid operation")
