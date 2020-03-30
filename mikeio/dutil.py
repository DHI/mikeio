import numpy as np
from System.Runtime.InteropServices import GCHandle, GCHandleType
import ctypes

from mikeio.eum import EUMType, EUMUnit, ItemInfo


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


def find_item(dfs, item_names):
    """Utility function to find item numbers

    Parameters
    ----------
    dfs : DfsFile

    item_names : list[str]
        Names of items to be found

    Returns
    -------
    list[int]
        item numbers (0-based)

    Raises
    ------
    KeyError
        In case item is not found in the dfs file
    """
    names = [x.Name for x in dfs.ItemInfo]
    item_lookup = {name: i for i, name in enumerate(names)}
    try:
        item_numbers = [item_lookup[x] for x in item_names]
    except KeyError:
        raise KeyError(f"Selected item name not found. Valid names are {names}")

    return item_numbers


def get_item_info(dfs, item_numbers):
    items = []
    for item in item_numbers:
        name = dfs.ItemInfo[item].Name
        eumItem = dfs.ItemInfo[item].Quantity.Item
        eumUnit = dfs.ItemInfo[item].Quantity.Unit
        itemtype = EUMType(eumItem)
        unit = EUMUnit(eumUnit)
        item = ItemInfo(name, itemtype, unit)
        items.append(item)
    return items


class Dataset:
    """Dataset

    Attributes
    ----------
    data: list[np.array]
        Data, potentially multivariate and multiple spatial dimensions
    time: list[datetime]
        Datetime of each timestep
    names: list[str]
        Names of each item in the data list

    Notes
    -----
    Data from a specific item can be accessed using the name of the item
    similar to a dictionary.

    Attributes data, time, names can also be unpacked like a tuple

    Examples
    --------
    >>> ds = mikeio.read("tests/testdata/random.dfs0")
    >>> ds
    DataSet(data, time, items)
    Number of items: 2
    Shape: (1000,)
    2017-01-01 00:00:00 - 2017-07-28 03:00:00
    >>> ds.items
    ['VarFun01', 'NotFun']
    >>> ds['NotFun'][0:5]
    array([0.64048636, 0.65325695, nan, 0.21420799, 0.99915695])

    >> data,time,items = ds
    >> items
    ['VarFun01', 'NotFun']
    """

    def __init__(self, data, time, items):
        self.data = data
        self.time = time
        self.items = items

    def __repr__(self):
        n_items = len(self.items)

        out = []
        out.append("DataSet(data, time, items)")
        out.append(f"Number of items: {n_items}")
        out.append(f"Shape: {self.data[0].shape}")
        out.append(f"{self.time[0]} - {self.time[-1]}")

        return str.join("\n", out)

    def __len__(self):
        return 3  # [data,time,items]

    def __getitem__(self, x):

        if isinstance(x, int):
            if x == 0:
                return self.data
            if x == 1:
                return self.time
            if x == 2:
                return self.items

            if x > 2:
                raise IndexError("")

        if isinstance(x, str):
            item_lookup = {item.name: i for i, item in enumerate(self.items)}
            x = item_lookup[x]
            return self.data[x]

        raise Exception("Invalid operation")

    def _ipython_key_completions_(self):
        return self.names
