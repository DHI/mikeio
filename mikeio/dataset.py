import os
from datetime import datetime
from typing import Iterable, Sequence, Union, Mapping, Optional
import warnings
import numpy as np
import pandas as pd
from copy import deepcopy

import collections.abc

from mikecore.DfsFile import DfsSimpleType

from .eum import EUMType, ItemInfo
from .data_utils import DataUtilsMixin
from .spatial.FM_geometry import GeometryFM, GeometryFMLayered
from .base import TimeSeries
from .dataarray import DataArray
from .spatial.geometry import (
    _Geometry,
    GeometryPoint2D,
    GeometryPoint3D,
    GeometryUndefined,
)
from .spatial.grid_geometry import Grid1D, Grid2D


def _repeat_items(
    items_in: Sequence[ItemInfo], prefixes: Sequence[str]
) -> Sequence[ItemInfo]:
    """Rereat a list of items n times with different prefixes"""
    new_items = []
    for item_in in items_in:
        for prefix in prefixes:
            item = deepcopy(item_in)
            item.name = f"{prefix}, {item.name}"
            new_items.append(item)

    return new_items


class _DatasetPlotter:
    def __init__(self, ds: "Dataset") -> None:
        self.ds = ds

    def __call__(self, ax=None, figsize=None, **kwargs):

        if self.ds.dims == ("time",):
            df = self.ds.to_dataframe()
            df.plot(figsize=figsize, **kwargs)  # TODO ax

        # fig, ax = self._get_fig_ax(ax, figsize)

    @staticmethod
    def _get_fig_ax(ax=None, figsize=None):
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = plt.gcf()
        return fig, ax

    def scatter(self, x, y, ax=None, figsize=None, **kwargs):
        _, ax = self._get_fig_ax(ax, figsize)
        if "title" in kwargs:
            title = kwargs.pop("title")
            ax.set_title(title)
        xval = self.ds[x].values.ravel()
        yval = self.ds[y].values.ravel()
        ax.scatter(xval, yval, **kwargs)

        x = self.ds.items[x].name if isinstance(x, int) else x
        y = self.ds.items[y].name if isinstance(y, int) else y
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        return ax


class Dataset(DataUtilsMixin, TimeSeries, collections.abc.MutableMapping):

    deletevalue = 1.0e-35

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
    <mikeio.Dataset>
    Dimensions: (1000,)
    Time: 2017-01-01 00:00:00 - 2017-07-28 03:00:00
    Items:
      0:  VarFun01 <Water Level> (meter)
      1:  NotFun <Water Level> (meter)
    >>> ds['NotFun'][0:5]
    array([0.64048636, 0.65325695, nan, 0.21420799, 0.99915695])
    >>> ds = mikeio.read("tests/testdata/HD2D.dfsu")
    <mikeio.Dataset>
    Dimensions: (9, 884)
    Time: 1985-08-06 07:00:00 - 1985-08-07 03:00:00
    Items:
      0:  Surface elevation <Surface Elevation> (meter)
      1:  U velocity <u velocity component> (meter per sec)
      2:  V velocity <v velocity component> (meter per sec)
      3:  Current speed <Current Speed> (meter per sec)
    >>> ds2 = ds[['Surface elevation','Current speed']] # item selection by name
    >>> ds2
    <mikeio.Dataset>
    Dimensions: (9, 884)
    Time: 1985-08-06 07:00:00 - 1985-08-07 03:00:00
    Items:
      0:  Surface elevation <Surface Elevation> (meter)
      1:  Current speed <Current Speed> (meter per sec)
    >>> ds3 = ds2.isel([0,1,2], axis=0) # temporal selection
    >>> ds3
    <mikeio.Dataset>
    Dimensions: (3, 884)
    Time: 1985-08-06 07:00:00 - 1985-08-06 12:00:00
    Items:
      0:  Surface elevation <Surface Elevation> (meter)
      1:  Current speed <Current Speed> (meter per sec)
    >>> ds4 = ds3.isel([100,200], axis=1) # element selection
    >>> ds4
    <mikeio.Dataset>
    Dimensions: (3, 2)
    Time: 1985-08-06 07:00:00 - 1985-08-06 12:00:00
    Items:
      0:  Surface elevation <Surface Elevation> (meter)
      1:  Current speed <Current Speed> (meter per sec)
    >>>  ds5 = ds[[1,0]] # item selection by position
    >>>  ds5
    <mikeio.Dataset>
    Dimensions: (1000,)
    Time: 2017-01-01 00:00:00 - 2017-07-28 03:00:00
    Items:
      0:  NotFun <Water Level> (meter)
      1:  VarFun01 <Water Level> (meter)
    """

    def __init__(
        self,
        data: Union[Mapping[str, DataArray], Iterable[DataArray]],
        time=None,
        items=None,
        geometry: _Geometry = None,
        zn=None,
        dims=None,
    ):
        if self._is_DataArrays(data):
            validate = True
        else:
            data = self._create_dataarrays(
                data=data, time=time, items=items, geometry=geometry, zn=zn, dims=dims
            )
            validate = False
        return self._init_from_DataArrays(data, validate=validate)

    @staticmethod
    def _is_DataArrays(data):
        """Check if input is Sequence/Mapping of DataArrays"""
        if isinstance(data, (Dataset, DataArray)):
            return True
        if isinstance(data, Mapping):
            for da in data.values():
                if not isinstance(da, DataArray):
                    raise TypeError("Please provide List/Mapping of DataArrays")
            return True
        if isinstance(data, Iterable):
            for da in data:
                if not isinstance(da, DataArray):
                    return False
                    # raise TypeError("Please provide List/Mapping of DataArrays")
            return True
        return False

    @staticmethod
    def _create_dataarrays(
        data: Sequence[np.ndarray],
        time=None,
        items=None,
        geometry: _Geometry = None,
        zn=None,
        dims=None,
    ):
        if not isinstance(data, Iterable):
            data = [data]
        items = Dataset._parse_items(items, len(data))

        # TODO: skip validation for all items after the first?
        data_vars = {}
        for dd, it in zip(data, items):
            data_vars[it.name] = DataArray(
                data=dd, time=time, item=it, geometry=geometry, zn=zn, dims=dims
            )
        return data_vars

    def _init_from_DataArrays(self, data, validate=True):
        """Initialize Dataset object with Iterable of DataArrays"""
        self._data_vars = self._DataArrays_as_mapping(data)

        if (len(self) > 1) and validate:
            first = self[0]
            for da in self[1:]:
                first._is_compatible(da, raise_error=True)

        self._check_all_different_ids(self._data_vars.values())

        self.__itemattr = []
        for key, value in self._data_vars.items():
            self._set_name_attr(key, value)

        if len(self.items) > 1:
            self.plot = _DatasetPlotter(self)

        # since Dataset is MutableMapping it has values and keys by default
        # but we delete those to avoid confusion
        self.values = None
        self.keys = None

    # remove values and keys from dir to avoid confusion
    def __dir__(self):
        keys = sorted(list(super().__dir__()) + list(self.__dict__.keys()))
        return set([d for d in keys if d not in ("values", "keys")])

    @staticmethod
    def _parse_items(items, n_items_data):
        if items is None:
            # default Undefined items
            item_infos = [ItemInfo(f"Item_{j+1}") for j in range(n_items_data)]
        else:
            if len(items) != n_items_data:
                raise ValueError(
                    f"Number of items ({len(items)}) must match len of data ({n_items_data})"
                )

            item_infos = []
            for item in items:
                if isinstance(item, (EUMType, str)):
                    item = ItemInfo(item)
                elif not isinstance(item, ItemInfo):
                    raise TypeError(f"items of type: {type(item)} is not supported")
                # TODO: item.name = self._to_safe_name(item.name)
                item_infos.append(item)

            item_names = [it.name for it in item_infos]
            if len(set(item_names)) != len(item_names):
                raise ValueError(f"Item names must be unique ({item_names})!")

        return item_infos

    @staticmethod
    def _DataArrays_as_mapping(data):
        """Create dict of DataArrays if necessary"""
        if isinstance(data, Mapping):
            if isinstance(data, Dataset):
                return data
            data = Dataset._validate_item_names_and_keys(data)
            _ = Dataset._unique_item_names(data.values())
            return data

        if isinstance(data, DataArray):
            data = [data]

        item_names = Dataset._unique_item_names(data)

        data_map = {}
        for n, da in zip(item_names, data):
            data_map[n] = da
        return data_map

    @staticmethod
    def _validate_item_names_and_keys(data_map: Mapping[str, DataArray]):
        for key, da in data_map.items():
            if da.name == "NoName":
                da.name = key
            elif da.name != key:
                warnings.warn(
                    f"The key {key} does not match the item name ({da.name}) of the corresponding DataArray. Item name will be replaced with key."
                )
                da.name == key
        return data_map

    @staticmethod
    def _unique_item_names(das: Sequence[DataArray]):
        item_names = [da.name for da in das]
        if len(set(item_names)) != len(item_names):
            raise ValueError(
                f"Item names must be unique! ({item_names}). Please rename before constructing Dataset."
            )
            # TODO: make a list of unique items names
        return item_names

    @staticmethod
    def _check_all_different_ids(das):
        """Are all the DataArrays different objects or are some referring to the same"""
        for j, da1 in enumerate(das):
            for k, da2 in enumerate(das):
                if j != k:
                    Dataset._id_of_DataArrays_equal(da1, da2)

    @staticmethod
    def _id_of_DataArrays_equal(da1, da2):
        """Check if two DataArrays are actually the same object"""
        if id(da1) == id(da2):
            raise ValueError(
                f"Cannot add the same object ({da1.name}) twice! Create a copy first."
            )
        if id(da1.values) == id(da2.values):
            raise ValueError(
                f"DataArrays {da1.name} and {da2.name} refer to the same data! Create a copy first."
            )

    def _check_already_present(self, new_da):
        """Is the DataArray already present in the Dataset?"""
        for da in self:
            self._id_of_DataArrays_equal(da, new_da)

    # ---- end of init ---------

    @property
    def time(self):
        return list(self)[0].time

    @time.setter
    def time(self, new_time):
        new_time = self._parse_time(new_time)
        if len(self.time) != len(new_time):
            raise ValueError("Length of new time is wrong")
        for da in self:
            da.time = new_time

    @property
    def start_time(self):
        """First time instance (as datetime)"""
        return self.time[0].to_pydatetime()

    @property
    def end_time(self):
        """Last time instance (as datetime)"""
        return self.time[-1].to_pydatetime()

    @property
    def timestep(self):
        """Time step in seconds if equidistant (and at
        least two time instances); otherwise None
        """
        dt = None
        if len(self.time) > 1:
            if self.is_equidistant:
                dt = (self.time[1] - self.time[0]).total_seconds()
        return dt

    @property
    def is_equidistant(self):
        """Is Dataset equidistant in time?"""
        if len(self.time) < 3:
            return True
        return len(self.time.to_series().diff().dropna().unique()) == 1

    @property
    def data(self) -> Sequence[np.ndarray]:
        """Data as list of numpy arrays"""
        warnings.warn(
            "property data is deprecated",
            FutureWarning,
        )
        return [x.to_numpy() for x in self]

    @property
    def n_timesteps(self):
        """Number of time steps"""
        return len(self.time)

    @property
    def items(self):
        """ItemInfo for each of the DataArrays as a list"""
        return [x.item for x in self]

    @property
    def names(self):
        """Name of each of the DataArrays as a list"""
        return [da.name for da in self]

    @property
    def n_items(self):
        """Number of items/DataArrays, equivalent to len()"""
        return len(self._data_vars)

    @property
    def ndim(self):
        """Number of data dimensions of each DataArray"""
        return self[0].ndim

    @property
    def dims(self):
        """Named data dimensions of each DataArray"""
        return self[0].dims

    @property
    def shape(self):
        """Shape of each DataArray"""
        return self[0].shape

    # TODO: remove this
    @property
    def n_elements(self):
        """Number of spatial elements/points"""
        n_elem = np.prod(self.shape)
        if self.n_timesteps > 1:
            n_elem = int(n_elem / self.n_timesteps)
        return n_elem

    @property
    def deletevalue(self):
        """File delete value"""
        return self[0].deletevalue

    @property
    def geometry(self):
        """Geometry of each DataArray"""
        return list(self)[0].geometry

    @property
    def _zn(self):
        return list(self)[0]._zn

    def __repr__(self):
        if len(self) == 0:
            return "Empty <mikeio.Dataset>"

        out = ["<mikeio.Dataset>"]

        gtxt = list(self)[0]._geometry_txt()
        if gtxt:
            out.append(gtxt)

        dims = [f"{self.dims[i]}:{self.shape[i]}" for i in range(self.ndim)]
        dimsstr = ", ".join(dims)
        out.append(f"Dimensions: ({dimsstr})")

        timetxt = (
            f"Time: {self.time[0]} (time-invariant)"
            if self.n_timesteps == 1
            else f"Time: {self.time[0]} - {self.time[-1]} ({self.n_timesteps} records)"
        )
        out.append(timetxt)

        if not self.is_equidistant:
            out.append("-- Non-equidistant calendar axis --")
        if self.n_items > 10:
            out.append(f"Number of items: {self.n_items}")
        else:
            out.append("Items:")
            for i, item in enumerate(self.items):
                out.append(f"  {i}:  {item}")

        return str.join("\n", out)

    def __len__(self):
        return len(self._data_vars)

    def __iter__(self):
        yield from self._data_vars.values()

    def __setitem__(self, key, value):
        self.__set_or_insert_item(key, value, insert=False)

    def __set_or_insert_item(self, key, value, insert=False):
        if not isinstance(value, DataArray):
            try:
                value = DataArray(value)
                # TODO: warn that this is not the preferred way!
            except:
                raise ValueError("Input could not be interpreted as a DataArray")

        if len(self) > 0:
            self[0]._is_compatible(value)

        item_name = value.name

        if isinstance(key, int):
            is_replacement = not insert
            if is_replacement:
                key_str = self.names[key]
                self._data_vars[key_str] = value
            else:
                self._check_already_present(value)

                if item_name in self.names:
                    raise ValueError(
                        f"Item name {item_name} already in Dataset ({self.names})"
                    )
                all_keys = list(self._data_vars.keys())
                all_keys.insert(key, item_name)

                data_vars = {}
                for k in all_keys:
                    if k in self._data_vars.keys():
                        data_vars[k] = self._data_vars[k]
                    else:
                        data_vars[k] = value
                self._data_vars = data_vars

            self._set_name_attr(item_name, value)
        else:
            is_replacement = key in self.names
            if key != item_name:
                # TODO: what would be best in this situation?
                warnings.warn(
                    f"key '{key}' and item name '{item_name}' mismatch! item name will be replaced with key!"
                )
                value.name = key
            if not is_replacement:
                self._check_already_present(value)
            self._data_vars[key] = value
            self._set_name_attr(key, value)

        if len(self) == 2 and not is_replacement:
            # now big enough for a plotter
            self.plot = _DatasetPlotter(self)

    def insert(self, key: int, value: DataArray):
        """Insert DataArray in a specific position


        Parameters
        ----------
        key : int
            index in Dataset where DataArray should be inserted
        value : DataArray
            DataArray to be inserted, must comform with with existing DataArrays
            and must have a unique item name
        """
        self.__set_or_insert_item(key, value, insert=True)

        if isinstance(key, slice):
            s = self.time.slice_indexer(key.start, key.stop)
            time_steps = list(range(s.start, s.stop))
            return self.isel(time_steps, axis=0)

    def remove(self, key: Union[int, str]):
        """Remove DataArray from Dataset

        Parameters
        ----------
        key : int, str
            index or name of DataArray to be remove from Dataset

        See also
        --------
        pop
        """
        self.__delitem__(key)

    def popitem(self):
        """Pop first DataArray from Dataset

        See also
        --------
        pop
        """
        return self.pop(0)

    def _set_name_attr(self, name: str, value: DataArray):
        name = self._to_safe_name(name)
        item_names = [self._to_safe_name(n) for n in self.names]
        if (name not in item_names) and hasattr(self, name):
            # oh-no the item_name matches the name of another attr
            pass
        else:
            if name not in self.__itemattr:
                self.__itemattr.append(name)  # keep track of what we insert
            setattr(self, name, value)

    def _del_name_attr(self, name: str):
        name = self._to_safe_name(name)
        if name in self.__itemattr:
            self.__itemattr.remove(name)
            delattr(self, name)

    def __getitem__(self, key) -> Union[DataArray, "Dataset"]:

        # select time steps
        if isinstance(key, pd.DatetimeIndex) or self._is_key_time(key):
            time_steps = pd.Series(range(len(self.time)), index=self.time)[key]
            time_steps = (
                [time_steps] if np.isscalar(time_steps) else time_steps.to_numpy()
            )
            return self.isel(time_steps, axis=0)
        if isinstance(key, slice):
            if self._is_slice_time_slice(key):
                try:
                    s = self.time.slice_indexer(key.start, key.stop)
                    time_steps = list(range(s.start, s.stop))
                except:
                    time_steps = list(range(*key.indices(len(self.time))))
                return self.isel(time_steps, axis=0)

        # select items
        key = self._key_to_str(key)
        if isinstance(key, str):
            return self._data_vars[key]

        if isinstance(key, Iterable):
            data_vars = {}
            for v in key:
                data_vars[v] = self._data_vars[v]
            return Dataset(data_vars)

        raise TypeError(f"indexing with a {type(key)} is not (yet) supported")

    def _is_slice_time_slice(self, s):
        if (s.start is None) and (s.stop is None):
            return False
        if s.start is not None:
            if not self._is_key_time(s.start):
                return False
        if s.stop is not None:
            if not self._is_key_time(s.stop):
                return False
        return True

    def _is_key_time(self, key):
        if isinstance(key, str) and key in self.names:
            return False
        if isinstance(key, str) and len(key) > 0 and key[0].isnumeric():
            # TODO: try to parse with pandas
            return True
        if isinstance(key, (datetime, np.datetime64, pd.Timestamp)):
            return True
        return False

    def _key_to_str(self, key):

        if isinstance(key, str):
            return key
        if isinstance(key, int):
            return list(self._data_vars.keys())[key]
        if isinstance(key, slice):
            s = key.indices(len(self))
            return self._key_to_str(list(range(*s)))
        if isinstance(key, Iterable):
            keys = []
            for k in key:
                keys.append(self._key_to_str(k))
            return keys
        if hasattr(key, "name"):
            return key.name
        raise TypeError(f"indexing with type {type(key)} is not supported")

    def __delitem__(self, key):

        key = self._key_to_str(key)
        self._data_vars.__delitem__(key)
        self._del_name_attr(key)

    def rename(self, mapper: Mapping[str, str], inplace=False):

        if inplace:
            ds = self
        else:
            ds = self.copy()

        for old_name, new_name in mapper.items():
            da = ds._data_vars.pop(old_name)
            da.name = new_name
            ds._data_vars[new_name] = da
            self._del_name_attr(old_name)
            self._set_name_attr(new_name, da)

        return ds

    # ---- arithmetic ---------

    def __radd__(self, other):
        return self.__add__(other)

    def __add__(self, other):
        if isinstance(other, self.__class__):
            return self._add_dataset(other)
        else:
            return self._add_value(other)

    def __rsub__(self, other):
        ds = self.__mul__(-1.0)
        return other + ds

    def __sub__(self, other):
        if isinstance(other, self.__class__):
            return self._add_dataset(other, sign=-1.0)
        else:
            return self._add_value(-other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            raise ValueError("Multiplication is not possible for two Datasets")
        else:
            return self._multiply_value(other)

    def _add_dataset(self, other, sign=1.0):
        self._check_datasets_match(other)
        try:
            data = [
                self[x].to_numpy() + sign * other[y].to_numpy()
                for x, y in zip(self.items, other.items)
            ]
        except:
            raise ValueError("Could not add data in Dataset")
        time = self.time.copy()
        items = deepcopy(self.items)
        return Dataset(data, time, items)

    def _check_datasets_match(self, other):
        if self.n_items != other.n_items:
            raise ValueError(
                f"Number of items must match ({self.n_items} and {other.n_items})"
            )
        for j in range(self.n_items):
            if self.items[j].type != other.items[j].type:
                raise ValueError(
                    f"Item types must match. Item {j}: {self.items[j].type} != {other.items[j].type}"
                )
            if self.items[j].unit != other.items[j].unit:
                raise ValueError(
                    f"Item units must match. Item {j}: {self.items[j].unit} != {other.items[j].unit}"
                )
        if not np.all(self.time == other.time):
            raise ValueError("All timesteps must match")
        if self.shape != other.shape:
            raise ValueError("shape must match")

    def _add_value(self, value):
        try:
            data = [value + self[x].to_numpy() for x in self.items]
        except:
            raise ValueError(f"{value} could not be added to Dataset")
        items = deepcopy(self.items)
        time = self.time.copy()
        return Dataset(data, time, items)

    def _multiply_value(self, value):
        try:
            data = [value * self[x].to_numpy() for x in self.items]
        except:
            raise ValueError(f"{value} could not be multiplied to Dataset")
        items = deepcopy(self.items)
        time = self.time.copy()
        return Dataset(data, time, items)

    def describe(self, **kwargs):
        """Generate descriptive statistics by wrapping pandas describe()"""
        all_df = [
            pd.DataFrame(self.data[j].ravel(), columns=[self.items[j].name]).describe(
                **kwargs
            )
            for j in range(self.n_items)
        ]
        return pd.concat(all_df, axis=1)

    def copy(self):
        """Returns a copy of this dataset."""

        return deepcopy(self)

    def to_numpy(self):
        """Stack data to a single ndarray with shape (n_items, n_timesteps, ...)

        Note: the output will be

        Returns
        -------
        np.ndarray
        """
        return np.stack(self.data)

    @classmethod
    def combine(cls, *datasets):
        """Combine n Datasets either along items or time axis

        Parameters
        ----------
            *datasets: datasets to combine

        Returns
        -------
        Dataset
            a combined dataset

        Examples
        --------
        >>> import mikeio
        >>> from mikeio import Dataset
        >>> ds1 = mikeio.read("HD2D.dfsu", items=0)
        >>> ds1
        <mikeio.Dataset>
        Dimensions: (9, 884)
        Time: 1985-08-06 07:00:00 - 1985-08-07 03:00:00
        Items:
        0:  Surface elevation <Surface Elevation> (meter)
        >>> ds2 = mikeio.read("HD2D.dfsu", items=[2,3])
        >>> ds2
        <mikeio.Dataset>
        Dimensions: (9, 884)
        Time: 1985-08-06 07:00:00 - 1985-08-07 03:00:00
        Items:
        0:  V velocity <v velocity component> (meter per sec)
        1:  Current speed <Current Speed> (meter per sec)
        >>> ds3 = Dataset.combine(ds1,ds2)
        >>> ds3
        <mikeio.Dataset>
        Dimensions: (9, 884)
        Time: 1985-08-06 07:00:00 - 1985-08-07 03:00:00
        Items:
        0:  Surface elevation <Surface Elevation> (meter)
        1:  V velocity <v velocity component> (meter per sec)
        2:  Current speed <Current Speed> (meter per sec)
        """

        if isinstance(datasets[0], Iterable):
            if isinstance(datasets[0][0], Dataset):  # (Dataset, DataArray)):
                datasets = datasets[0]

        # if isinstance(datasets[0], DataArray):
        #     ds = datasets[0]._to_dataset()
        #     print("to dataset")
        # else:
        ds = datasets[0].copy()

        for dsj in datasets[1:]:
            ds = ds._combine(dsj, copy=False)
        return ds

    def _combine(self, other, copy=True):
        try:
            ds = self._concat_time(other, copy=copy)
        except ValueError:
            ds = self._append_items(other, copy=copy)
        return ds

    def append(self, other, inplace=False):
        # TODO: require other da
        return self.append_items(other, inplace)

    def append_items(self, other, inplace=False):
        """Append items from other Dataset to this Dataset"""
        if inplace:
            self._append_items(other, copy=False)
        else:
            return self._append_items(other, copy=True)

    def _append_items(self, other, copy=True):
        if isinstance(other, DataArray):
            other = other._to_dataset()
        item_names = {item.name for item in self.items}
        other_names = {item.name for item in other.items}

        overlap = other_names.intersection(item_names)
        if len(overlap) != 0:
            raise ValueError("Can not append items, names are not unique")

        if not np.all(self.time == other.time):
            # if not: create common time?
            raise ValueError("All timesteps must match")
        ds = self.copy() if copy else self

        for key, value in other._data_vars.items():
            if key != "Z coordinate":
                ds[key] = value

        return ds

    def concat(self, other):
        """Concatenate this Dataset with data from other Dataset

        Parameters
        ---------
        other: Dataset
            Other dataset to concatenate with

        Returns
        -------
        Dataset
            concatenated dataset


        Examples
        --------
        >>> import mikeio
        >>> ds1 = mikeio.read("HD2D.dfsu", time_steps=[0,1])
        >>> ds2 = mikeio.read("HD2D.dfsu", time_steps=[2,3])
        >>> ds1.n_timesteps
        2
        >>> ds3 = ds1.concat(ds2)
        >>> ds3.n_timesteps
        4
        """

        ds = self._concat_time(other, copy=True)

        return ds

    def _concat_time(self, other, copy=True):
        self._check_all_items_match(other)
        if not np.all(self.shape[1:] == other.shape[1:]):
            raise ValueError("Shape of the datasets must match (except time dimension)")
        ds = self.copy() if copy else self

        s1 = pd.Series(np.arange(len(ds.time)), index=ds.time, name="idx1")
        s2 = pd.Series(np.arange(len(other.time)), index=other.time, name="idx2")
        df12 = pd.concat([s1, s2], axis=1)

        newtime = df12.index
        newdata = self.create_empty_data(
            n_items=ds.n_items, n_timesteps=len(newtime), shape=ds.shape[1:]
        )
        idx1 = np.where(~df12["idx1"].isna())
        idx2 = np.where(~df12["idx2"].isna())
        for j in range(ds.n_items):
            # if there is an overlap "other" data will be used!
            newdata[j][idx1, :] = ds.data[j]
            newdata[j][idx2, :] = other.data[j]

        return Dataset(newdata, newtime, ds.items)

    def _check_all_items_match(self, other):
        if self.n_items != other.n_items:
            raise ValueError(
                f"Number of items must match ({self.n_items} and {other.n_items})"
            )
        for j in range(self.n_items):
            if self.items[j].name != other.items[j].name:
                raise ValueError(
                    f"Item names must match. Item {j}: {self.items[j].name} != {other.items[j].name}"
                )
            if self.items[j].type != other.items[j].type:
                raise ValueError(
                    f"Item types must match. Item {j}: {self.items[j].type} != {other.items[j].type}"
                )
            if self.items[j].unit != other.items[j].unit:
                raise ValueError(
                    f"Item units must match. Item {j}: {self.items[j].unit} != {other.items[j].unit}"
                )

    def dropna(self):
        """Remove time steps where all items are NaN"""

        # TODO consider all items
        x = self[0].to_numpy()

        # this seems overly complicated...
        axes = tuple(range(1, x.ndim))
        idx = np.where(~np.isnan(x).all(axis=axes))
        idx = list(idx[0])

        return self.isel(idx, axis=0)

    def flipud(self):
        """Flip dataset upside down"""
        self._data_vars = {
            key: value.flipud() for (key, value) in self._data_vars.items()
        }
        return self

    # ===== select =========

    def isel(self, idx=None, axis=0, **kwargs):
        """
        Select subset along an axis.

        Parameters
        ----------
        idx: int, scalar or array_like
        axis: (int, str, None), optional
            axis number or "time", by default 1

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
        res = [da.isel(idx=idx, axis=axis, **kwargs) for da in self]

        return Dataset(res)

    # ===== aggregate =========

    def aggregate(self, axis="time", func=np.nanmean, **kwargs):
        """Aggregate along an axis

        Parameters
        ----------
        axis: (int, str, None), optional
            axis number or "time" or "space", by default "time"=0
        func: function, optional
            default np.nanmean

        Returns
        -------
        Dataset
            dataset with aggregated values
        """

        res = {
            name: da.aggregate(axis=axis, func=func, **kwargs)
            for name, da in self._data_vars.items()
        }

        return Dataset(res)

    def quantile(self, q, *, axis="time", **kwargs):
        """Compute the q-th quantile of the data along the specified axis.

        Wrapping np.quantile

        Parameters
        ----------
        q: array_like of float
            Quantile or sequence of quantiles to compute,
            which must be between 0 and 1 inclusive.
        axis: (int, str, None), optional
            axis number or "time" or "space", by default "time"=0

        Returns
        -------
        Dataset
            dataset with quantile values

        Examples
        --------
        >>> ds.quantile(q=[0.25,0.75])
        >>> ds.quantile(q=0.5)
        >>> ds.quantile(q=[0.01,0.5,0.99], axis="space")

        See Also
        --------
        nanquantile : quantile with NaN values ignored
        """
        return self._quantile(q, axis=axis, func=np.quantile, **kwargs)

    def nanquantile(self, q, *, axis="time", **kwargs):
        """Compute the q-th quantile of the data along the specified axis, while ignoring nan values.

        Wrapping np.nanquantile

        Parameters
        ----------
        q: array_like of float
            Quantile or sequence of quantiles to compute,
            which must be between 0 and 1 inclusive.
        axis: (int, str, None), optional
            axis number or "time" or "space", by default "time"=0

        Examples
        --------
        >>> ds.nanquantile(q=[0.25,0.75])
        >>> ds.nanquantile(q=0.5)
        >>> ds.nanquantile(q=[0.01,0.5,0.99], axis="space")

        Returns
        -------
        Dataset
            dataset with quantile values
        """
        return self._quantile(q, axis=axis, func=np.nanquantile, **kwargs)

    def _quantile(self, q, *, axis=0, func=np.quantile, **kwargs):

        if np.isscalar(q):
            res = [da._quantile(q=q, axis=axis, func=func) for da in self]
        else:
            res = []

            for name, da in self._data_vars.items():
                for quantile in q:
                    qd = da._quantile(q=quantile, axis=axis, func=func)
                    newname = f"Quantile {quantile}, {name}"
                    qd.name = newname
                    res.append(qd)

        return Dataset(res)

    def max(self, axis="time"):
        """Max value along an axis

        Parameters
        ----------
        axis: (int, str, None), optional
            axis number or "time" or "space", by default "time"=0

        Returns
        -------
        Dataset
            dataset with max value

        See Also
        --------
            nanmax : Max values with NaN values removed
        """
        return self.aggregate(axis=axis, func=np.max)

    def min(self, axis="time"):
        """Min value along an axis

        Parameters
        ----------
        axis: (int, str, None), optional
            axis number or "time" or "space", by default "time"=0

        Returns
        -------
        Dataset
            dataset with max value

        See Also
        --------
            nanmin : Min values with NaN values removed
        """
        return self.aggregate(axis=axis, func=np.min)

    def mean(self, axis="time"):
        """Mean value along an axis

        Parameters
        ----------
        axis: (int, str, None), optional
            axis number or "time" or "space", by default "time"=0

        Returns
        -------
        Dataset
            dataset with mean value

        See Also
        --------
            nanmean : Mean values with NaN values removed
            average: Weighted average
        """
        return self.aggregate(axis=axis, func=np.mean)

    def average(self, weights, axis="time"):
        """
        Compute the weighted average along the specified axis.

        Parameters
        ----------
        axis: (int, str, None), optional
            axis number or "time" or "space", by default "time"=0

        Returns
        -------
        Dataset
            dataset with weighted average value

        See Also
        --------
            nanmean : Mean values with NaN values removed
            aggregate: Weighted average

        Examples
        --------
        >>> dfs = Dfsu("HD2D.dfsu")
        >>> ds = dfs.read(["Current speed"])
        >>> area = dfs.get_element_area()
        >>> ds2 = ds.average(axis="space", weights=area)
        """

        def func(x, axis, keepdims):
            if keepdims:
                raise NotImplementedError()

            return np.average(x, weights=weights, axis=axis)

        return self.aggregate(axis=axis, func=func)

    def nanmax(self, axis="time"):
        """Max value along an axis (NaN removed)

        Parameters
        ----------
        axis: (int, str, None), optional
            axis number or "time" or "space", by default "time"=0

        Returns
        -------
        Dataset
            dataset with max value
        """
        return self.aggregate(axis=axis, func=np.nanmax)

    def nanmin(self, axis="time"):
        """Min value along an axis (NaN removed)

        Parameters
        ----------
        axis: (int, str, None), optional
            axis number or "time" or "space", by default "time"=0

        Returns
        -------
        Dataset
            dataset with max value
        """
        return self.aggregate(axis=axis, func=np.nanmin)

    def nanmean(self, axis="time"):
        """Mean value along an axis (NaN removed)

        Parameters
        ----------
        axis: (int, str, None), optional
            axis number or "time" or "space", by default "time"=0

        Returns
        -------
        Dataset
            dataset with mean value
        """
        return self.aggregate(axis=axis, func=np.nanmean)

    def squeeze(self):
        """
        Remove axes of length 1

        Returns
        -------
        Dataset
        """
        res = {name: da.squeeze() for name, da in self._data_vars.items()}

        return Dataset(res)

    def sel(
        self,
        *,
        time: Union[int, pd.DatetimeIndex, "Dataset"] = None,
        x: float = None,
        y: float = None,
        z: float = None,
        **kwargs,
    ) -> "Dataset":
        """
        Examples
        --------
        ds.sel(layer='bottom')
        ds.sel(x=1.0,y=55.0)
        ds.sel(area=[1.,12., 2., 15.])
        """

        # select in space
        if (x is not None) or (y is not None) or (z is not None):
            # idx = self.geometry.find_nearest_elements(x=x, y=y, z=z)
            idx = self.geometry.find_index(x=x, y=y, z=z)
            ds = self.isel(idx, axis="space")
        else:
            ds = self

        if "layer" in kwargs:
            if isinstance(ds.geometry, GeometryFMLayered):
                layer = kwargs.pop("layer")
                idx = ds.geometry.get_layer_elements(layer)
                ds = ds.isel(idx, axis="space")
            else:
                raise ValueError("'layer' can only be selected from layered Dfsu data")

        if "area" in kwargs:
            if isinstance(ds.geometry, GeometryFM):
                area = kwargs.pop("area")
                idx = ds.geometry._elements_in_area(area)
                ds = ds.isel(idx, axis="space")
            else:
                raise ValueError("'area' can only be selected from Dfsu data")

        if len(kwargs) > 0:
            args = ",".join(kwargs)
            raise ValueError(f"Argument(s) '{args}' not recognized (layer, area).")

        # select in time
        if time is not None:
            time = time.time if isinstance(time, TimeSeries) else time
            if isinstance(time, int) or (
                isinstance(time, Sequence) and isinstance(time[0], int)
            ):
                ds = ds.isel(time, axis="time")
            else:
                ds = ds[time]

        return ds

    def interp(
        self,
        *,
        time: Union[pd.DatetimeIndex, "DataArray"] = None,
        x: float = None,
        y: float = None,
        z: float = None,
        n_nearest=3,
        **kwargs,
    ) -> "Dataset":

        if z is not None:
            raise NotImplementedError()

        # interp in space
        if (x is not None) or (y is not None) or (z is not None):
            xy = [(x, y)]

            if isinstance(
                self.geometry, GeometryFM
            ):  # TODO remove this when all geometries implements the same method
                interpolant = self.geometry.get_2d_interpolant(
                    xy, n_nearest=n_nearest, **kwargs
                )
                das = [da.interp(x=x, y=y, interpolant=interpolant) for da in self]
            else:
                das = [da.interp(x=x, y=y) for da in self]
            ds = Dataset(das)
        else:
            ds = Dataset([da for da in self])

        # interp in time
        if time is not None:
            ds = ds.interp_time(time)

        return ds

    def interp_like(
        self,
        other: Union["Dataset", DataArray, Grid2D, GeometryFM, pd.DatetimeIndex],
        **kwargs,
    ) -> "Dataset":
        """Interpolate in space (and in time) to other geometry (and time axis)

        Note: currently only supports interpolation from dfsu-2d to
              dfs2 or other dfsu-2d Datasets

        Parameters
        ----------
        other: Dataset, DataArray, Grid2D, GeometryFM, pd.DatetimeIndex
        kwargs: additional kwargs are passed to interpolation method

        Examples
        --------
        >>> ds = mikeio.read("HD.dfsu")
        >>> ds2 = mikeio.read("wind.dfs2")
        >>> dsi = ds.interp_like(ds2)
        >>> dsi.to_dfs("HD_gridded.dfs2")
        >>> dse = ds.interp_like(ds2, extrapolate=True)
        >>> dst = ds.interp_like(ds2.time)

        Returns
        -------
        Dataset
            Interpolated Dataset
        """
        if isinstance(other, pd.DatetimeIndex):
            return self.interp_time(other, **kwargs)

        if hasattr(other, "geometry"):
            geom = other.geometry
        else:
            geom = other

        if isinstance(geom, Grid2D):
            xy = geom.xy

        elif isinstance(geom, GeometryFM):
            xy = geom.element_coordinates[:, :2]
            if geom.is_layered:
                raise NotImplementedError(
                    "Does not yet support layered flexible mesh data!"
                )
        else:
            raise NotImplementedError()

        interpolant = self.geometry.get_2d_interpolant(xy, **kwargs)
        das = [da.interp_like(geom, interpolant=interpolant) for da in self]
        ds = Dataset(das)

        if hasattr(other, "time"):
            ds = ds.interp_time(other.time)

        return ds

    def interp_time(
        self,
        dt: Union[float, pd.DatetimeIndex, "Dataset"],
        *,
        method="linear",
        extrapolate=True,
        fill_value=np.nan,
    ):
        """Temporal interpolation

        Wrapper of `scipy.interpolate.interp`

        Parameters
        ----------
        dt: float or pd.DatetimeIndex or Dataset
            output timestep in seconds
        method: str or int, optional
            Specifies the kind of interpolation as a string ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', 'next', where 'zero', 'slinear', 'quadratic' and 'cubic' refer to a spline interpolation of zeroth, first, second or third order; 'previous' and 'next' simply return the previous or next value of the point) or as an integer specifying the order of the spline interpolator to use. Default is 'linear'.
        extrapolate: bool, optional
            Default True. If False, a ValueError is raised any time interpolation is attempted on a value outside of the range of x (where extrapolation is necessary). If True, out of bounds values are assigned fill_value
        fill_value: float or array-like, optional
            Default NaN. this value will be used to fill in for points outside of the time range.

        Returns
        -------
        Dataset

        Examples
        --------
        >>> ds = mikeio.read("tests/testdata/HD2D.dfsu")
        >>> ds
        <mikeio.Dataset>
        Dimensions: (9, 884)
        Time: 1985-08-06 07:00:00 - 1985-08-07 03:00:00
        Items:
        0:  Surface elevation <Surface Elevation> (meter)
        1:  U velocity <u velocity component> (meter per sec)
        2:  V velocity <v velocity component> (meter per sec)
        3:  Current speed <Current Speed> (meter per sec)
        >>> dsi = ds.interp_time(dt=1800)
        >>> dsi
        <mikeio.Dataset>
        Dimensions: (41, 884)
        Time: 1985-08-06 07:00:00 - 1985-08-07 03:00:00
        Items:
        0:  Surface elevation <Surface Elevation> (meter)
        1:  U velocity <u velocity component> (meter per sec)
        2:  V velocity <v velocity component> (meter per sec)
        3:  Current speed <Current Speed> (meter per sec)
        """
        t_out_index = self._parse_interp_time(self.time, dt)
        t_in = self.time.values.astype(float)
        t_out = t_out_index.values.astype(float)

        # TODO: it would be more efficient to interp all data at once!
        data = [
            self._interpolate_time(
                t_in, t_out, da.to_numpy(), method, extrapolate, fill_value
            )
            for da in self
        ]

        zn = (
            None
            if self._zn is None
            else self._interpolate_time(
                t_in, t_out, self._zn, method, extrapolate, fill_value
            )
        )

        return Dataset(
            data,
            t_out_index,
            items=self.items.copy(),
            geometry=self.geometry,
            zn=zn,
        )

    def to_dataframe(self, unit_in_name=False, round_time="ms"):
        """Convert Dataset to a Pandas DataFrame

        Parameters
        ----------
        unit_in_name: bool, optional
            include unit in column name, default False

        Returns
        -------
        pd.DataFrame
        """

        if len(self.data[0].shape) != 1:
            self = self.squeeze()

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

        if round_time:
            rounded_idx = pd.DatetimeIndex(self.time).round(round_time)
            df.index = pd.DatetimeIndex(rounded_idx, freq="infer")
        else:
            df.index = pd.DatetimeIndex(self.time, freq="infer")

        return df

    def _ipython_key_completions_(self):
        return [x.name for x in self.items]

    @staticmethod
    def create_empty_data(n_items=1, n_timesteps=1, n_elements=None, shape=None):
        data = []
        if shape is None:
            if n_elements is None:
                raise ValueError("n_elements and shape cannot both be None")
            else:
                shape = n_elements
        if np.isscalar(shape):
            shape = [shape]
        dati = np.empty(shape=(n_timesteps, *shape))
        dati[:] = np.nan
        for _ in range(n_items):
            data.append(dati.copy())
        return data

    @staticmethod
    def _validate_extension(filename, valid_extension):
        _, ext = os.path.splitext(filename)
        if ext != valid_extension:
            raise ValueError(f"File extension must be {valid_extension}")

    def to_dfs(self, filename, **kwargs):

        filename = str(filename)

        if isinstance(
            self.geometry, (GeometryPoint2D, GeometryPoint3D, GeometryUndefined)
        ):
            if self.ndim == 1 and self.dims[0][0] == "t":
                self._validate_extension(filename, ".dfs0")
                self._to_dfs0(filename, **kwargs)
            else:
                raise ValueError("Cannot write Dataset with no geometry to file!")
        elif isinstance(self.geometry, Grid2D):
            self._validate_extension(filename, ".dfs2")
            self._to_dfs2(filename)

        elif isinstance(self.geometry, Grid1D):
            self._validate_extension(filename, ".dfs1")
            self._to_dfs1(filename)
        elif isinstance(self.geometry, GeometryFM):
            self._validate_extension(filename, ".dfsu")
            self._to_dfsu(filename)
        else:
            raise NotImplementedError(
                "Writing this type of dataset is not yet implemented"
            )

    def _to_dfs0(self, filename, **kwargs):
        from .dfs0 import _write_dfs0

        dtype = kwargs.get("dtype", DfsSimpleType.Float)

        _write_dfs0(filename, self, dtype=dtype)

    def _to_dfs2(self, filename):
        # assumes Grid2D geometry
        from .dfs2 import write_dfs2

        write_dfs2(filename, self)

    def _to_dfs1(self, filename):
        from .dfs1 import Dfs1

        dfs = Dfs1()
        dfs.write(filename, data=self, dx=self.geometry.dx, x0=self.geometry.x0)

    def _to_dfsu(self, filename):
        from .dfsu import _write_dfsu

        _write_dfsu(filename, self)
