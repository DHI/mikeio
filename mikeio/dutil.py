import numpy as np
import pandas as pd
from copy import deepcopy
from mikeio.eum import EUMType, EUMUnit, ItemInfo


def get_valid_items_and_timesteps(dfs, items, time_steps):
    # TODO consider if this should be part of a DFS base class

    if isinstance(items, int) or isinstance(items, str):
        items = [items]

    if items is not None and isinstance(items[0], str):
        items = find_item(dfs._source, items)

    if items is None:
        item_numbers = list(range(dfs._n_items))
    else:
        item_numbers = items

    if time_steps is None:
        time_steps = list(range(dfs._n_timesteps))

    if isinstance(time_steps, int):
        time_steps = [time_steps]

    items = get_item_info(dfs._source, item_numbers)

    return items, item_numbers, time_steps


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
    """Read DFS ItemInfo

    Parameters
    ----------
    dfs : MIKE dfs object
    item_numbers : list[int]
        
    Returns
    -------
    list[Iteminfo]
    """
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
    <mikeio.DataSet>
    Dimensions: (1000,)
    Time: 2017-01-01 00:00:00 - 2017-07-28 03:00:00
    Items:
      0:  VarFun01 <Water Level> (meter)
      1:  NotFun <Water Level> (meter)
    >>> ds['NotFun'][0:5]
    array([0.64048636, 0.65325695, nan, 0.21420799, 0.99915695])
    >>> ds = mikeio.read("tests/testdata/HD2D.dfsu")
    <mikeio.DataSet>
    Dimensions: (9, 884)
    Time: 1985-08-06 07:00:00 - 1985-08-07 03:00:00
    Items:
      0:  Surface elevation <Surface Elevation> (meter)
      1:  U velocity <u velocity component> (meter per sec)
      2:  V velocity <v velocity component> (meter per sec)
      3:  Current speed <Current Speed> (meter per sec)
    >>> ds2 = ds[['Surface elevation','Current speed']] # item selection by name
    >>> ds2
    <mikeio.DataSet>
    Dimensions: (9, 884)
    Time: 1985-08-06 07:00:00 - 1985-08-07 03:00:00
    Items:
      0:  Surface elevation <Surface Elevation> (meter)
      1:  Current speed <Current Speed> (meter per sec)
    >>> ds3 = ds2.isel([0,1,2], axis=0) # temporal selection
    >>> ds3
    <mikeio.DataSet>
    Dimensions: (3, 884)
    Time: 1985-08-06 07:00:00 - 1985-08-06 12:00:00
    Items:
      0:  Surface elevation <Surface Elevation> (meter)
      1:  Current speed <Current Speed> (meter per sec)
    >>> ds4 = ds3.isel([100,200], axis=1) # element selection
    >>> ds4
    <mikeio.DataSet>
    Dimensions: (3, 2)
    Time: 1985-08-06 07:00:00 - 1985-08-06 12:00:00
    Items:
      0:  Surface elevation <Surface Elevation> (meter)
      1:  Current speed <Current Speed> (meter per sec)
    >>>  ds5 = ds[[1,0]] # item selection by position
    >>>  ds5 
    <mikeio.DataSet>
    Dimensions: (1000,)
    Time: 2017-01-01 00:00:00 - 2017-07-28 03:00:00
    Items:
      0:  NotFun <Water Level> (meter)
      1:  VarFun01 <Water Level> (meter)
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
        self.time = pd.DatetimeIndex(time, freq="infer")
        self.items = items

    def __repr__(self):
        n_items = len(self.items)

        shape = self.data[0].shape
        if len(self) > 1 and self.items[0].name == "Z coordinate":
            shape = self.data[1].shape

        out = []
        out.append("<mikeio.DataSet>")
        out.append(f"Dimensions: {shape}")
        out.append(f"Time: {self.time[0]} - {self.time[-1]}")
        if n_items > 10:
            out.append(f"Number of items: {n_items}")
        else:
            out.append("Items:")
            for i, item in enumerate(self.items):
                out.append(f"  {i}:  {item}")

        return str.join("\n", out)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, x):

        if isinstance(x, int):
            return self.data[x]

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
                data_item = self.__getitem__(v)
                if isinstance(v, str):
                    i = item_lookup[v]
                if isinstance(v, int):
                    i = v

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

        items = self.items

        if axis == 1 and items[0].name == "Z coordinate":
            items = deepcopy(items)
            items.pop(0)

        time = self.time
        if axis == 0:
            time = time[idx]

        res = []
        for item in items:
            x = np.take(self[item.name], idx, axis=axis)
            res.append(x)

        ds = Dataset(res, time, items)
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
        return [x.name for x in self.items]

    @property
    def is_equidistant(self):
        return self.time.freq is not None
