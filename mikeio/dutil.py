import numpy as np
import pandas as pd
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
    items: list[ItemInfo]
        Names, type and unit of each item in the data list

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
    [VarFun01 <Water Level> (meter), NotFun <Water Level> (meter)]
    >>> ds['NotFun'][0:5]
    array([0.64048636, 0.65325695, nan, 0.21420799, 0.99915695])

    >> data,time,items = ds
    >> items
    [VarFun01 <Water Level> (meter), NotFun <Water Level> (meter)]
    """

    def __init__(self, data, time, items):

        n_items = len(data)
        n_timesteps = data[0].shape[0]

        if len(time) != n_timesteps:
            raise ValueError(
                f"Number of timesteps in time {len(time)} doesn't match the data {n_timesteps}."
            )
        if len(items) != n_items:
            raise ValueError(
                f"Number of items in iteminfo {len(items)} doesn't match the data {n_items}."
            )
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

    def isel(self, idx, axis=1):
        """
        Select subset along an axis.

        Parameters
        ----------
        idx: int, scalar or array_like
        axis: int, optional
            default 1

        Returns
        -------
        Dataset
            dataset with subset
        """

        if axis == 0:
            raise ValueError("Subsetting along time axis not supported")

        res = []
        for item in self.items:
            x = np.take(self[item.name], idx, axis=axis)
            res.append(x)

        ds = Dataset(res, self.time, self.items)
        return ds

    def to_dataframe(self, unit_in_name=False):
        """Convert Dataset to a Pandas DataFrame
        
        Parameters
        ----------
        filename: str
            full path and file name to the dfs0 file.
        unit_in_name: bool, optional
            include unit in column name, default False
        
        Returns
        -------
        pd.DataFrame
        """

        if len(self.data[0].shape) != 1:
            raise ValueError(
                "Only data with a single dimension can be converted to a dataframe. Hint: use `isel` to create a subset."
            )

        if unit_in_name:
            names = [f"{item.name} ({item.unit.name})" for item in self.items]
        else:
            names = [item.name for item in self.items]

        data = np.asarray(self.data).T
        df = pd.DataFrame(data, columns=names)

        df.index = pd.DatetimeIndex(self.time, freq="infer")

        return df

    def _ipython_key_completions_(self):
        return self.names
