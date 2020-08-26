import numpy as np
import pandas as pd
from mikeio.eum import EUMType, EUMUnit, ItemInfo

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
    >>> import mikeio
    >>> ds = mikeio.read("tests/testdata/HD2D.dfsu")
    >>> ds.items
    [Surface elevation <Surface Elevation> (meter), U velocity <u velocity component> (meter per sec), V velocity <v velocity component> (meter per sec), Current speed <Current Speed> (meter per sec)]
    >>> ds2 = ds[['Surface elevation','Current speed']] # item selection
    >>> ds2.items
    [Surface elevation <Surface Elevation> (meter), Current speed <Current Speed> (meter per sec)]
    >>> ds3 = ds2.isel([0,1,2], axis=0) # temporal selection
    >>> ds3
    DataSet(data, time, items)
    Number of items: 2
    Shape: (3, 884)
    1985-08-06 07:00:00 - 1985-08-06 12:00:00
    >>> ds4 = ds3.isel([100,200], axis=1) # element selection
    >>> ds4
    DataSet(data, time, items)
    Number of items: 2
    Shape: (3, 2)
    1985-08-06 07:00:00 - 1985-08-06 12:00:00
>>>
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
        self.time = pd.DatetimeIndex(time, freq='infer')
        self.items = items

    def __repr__(self):
        n_items = len(self.items)

        out = []
        out.append("<mikeio.DataSet>")
        out.append(f"Dimensions: {self.data[0].shape}")
        out.append(f"Time: {self.time[0]} - {self.time[-1]}")
        if n_items > 10:
            out.append(f"Number of items: {n_items}")
        else:
            out.append("Items:")
            for item in self.items:
                out.append(f"  {item}")
        
        

        return str.join("\n", out)

    def __len__(self):
        return len(self.items)

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

        if isinstance(x, ItemInfo):
            return self.__getitem__(x.name)

        if isinstance(x, list):
            data = []
            items = []

            item_lookup = {item.name: i for i, item in enumerate(self.items)}

            for v in x:
                data_item =  self.__getitem__(v)
                i = item_lookup[v]
                item = self.items[i]
                items.append(item)
                data.append(data_item)

            return Dataset(data, self.time, items)
        
        

        raise Exception("Invalid operation")

    def isel(self, idx, axis=1):
        """
        Select subset along an axis.

        Parameters
        ----------
        idx: int, scalar or array_like
        axis: int, optional
            default 1, 0= temporal axis

        Returns
        -------
        Dataset
            dataset with subset

        Examples
        --------
        >>> ds = mikeio.read("tests/testdata/HD2D.dfsu")
        >>> ds2 = ds.isel([0,1,2], axis=0) # temporal selection
        >>> ds2
        DataSet(data, time, items)
        Number of items: 2
        Shape: (3, 884)
        1985-08-06 07:00:00 - 1985-08-06 12:00:00
        >>> ds3 = ds2.isel([100,200], axis=1) # element selection
        >>> ds3
        DataSet(data, time, items)
        Number of items: 2
        Shape: (3, 2)
        1985-08-06 07:00:00 - 1985-08-06 12:00:00
        """

        time = self.time
        if axis == 0:
            time = time[idx]

        res = []
        for item in self.items:
            x = np.take(self[item.name], idx, axis=axis)
            res.append(x)

        ds = Dataset(res, time, self.items)
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
