from typing import Iterable, Sequence, Union, Mapping, Optional
import numpy as np
import pandas as pd
from copy import deepcopy
from mikeio.eum import EUMType, ItemInfo
import collections.abc

import mikeio.data_utils as du
from .base import TimeSeries
from .dataarray import DataArray
from .spatial.geometry import _Geometry
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

    # def __call__(self, ax=None, figsize=None, **kwargs):
    #     fig, ax = self._get_fig_ax(ax, figsize)

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


class Dataset(TimeSeries, collections.abc.MutableMapping):
    # TODO: Dataset(Mapping)

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

    # @staticmethod
    # def from_data_time_items(
    #     data: Union[Sequence[np.ndarray], float],
    #     time: Union[pd.DatetimeIndex, str],
    #     items: Union[Sequence[ItemInfo], Sequence[EUMType], Sequence[str]] = None,
    #     geometry: _Geometry = None,
    #     zn=None,
    #     dims: Optional[Sequence[str]] = None,
    # ):

    #     item_infos = []
    #     time = du._parse_time(time)

    #     if isinstance(data, Sequence) and hasattr(data[0], "shape"):
    #         n_items = len(data)
    #         n_timesteps = data[0].shape[0]
    #     else:
    #         raise TypeError(
    #             f"data type '{type(data)}' not supported! data must be a list of numpy arrays"
    #         )

    #     item_infos = Dataset._parse_items(items, n_items)

    #     if len(time) != n_timesteps:
    #         raise ValueError(
    #             f"Number of timesteps in time {len(time)} doesn't match the data {n_timesteps}."
    #         )
    #     time = pd.DatetimeIndex(time)

    #     data_vars = {}
    #     for dd, it in zip(data, item_infos):
    #         data_vars[it.name] = DataArray(
    #             data=dd, time=time, item=it, geometry=geometry, zn=zn, dims=dims
    #         )

    #     ds = Dataset(data_vars)

    #     return ds

    @staticmethod
    def _create_dataarrays(
        data: Sequence[np.ndarray],
        time=None,
        items=None,
        geometry: _Geometry = None,
        zn=None,
        dims=None,
    ):
        if not isinstance(data, Sequence):
            data = [data]
        items = Dataset._parse_items(items, len(data))

        # TODO: skip validation for all items after the first?
        data_vars = {}
        names = []
        for dd, it in zip(data, items):
            n = it.name  # TODO: du._to_safe_name(it.name)
            if n in names:
                raise ValueError()  # TODO: repeated item names!
            names.append(n)
            data_vars[n] = DataArray(
                data=dd, time=time, item=it, geometry=geometry, zn=zn, dims=dims
            )
        return data_vars

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
                # TODO: item.name = du._to_safe_name(item.name)
                item_infos.append(item)

        return item_infos

    @staticmethod
    def _DataArrays_as_mapping(data):
        """Create dict of DataArrays if necessary"""
        if isinstance(data, Mapping):
            # TODO: What if keys and item names does not match?
            return data
        else:
            assert isinstance(data[0], DataArray)

            item_names = []
            for da in data:
                item_names.append(du._to_safe_name(da.name))
            if len(set(item_names)) != len(item_names):
                raise ValueError(f"Item names must be unique! ({item_names})")
            # TODO: make a list of unique items names

            data_map = {}
            for n, da in zip(item_names, data):
                data_map[n] = da
            return data_map

    def _init_from_DataArrays(self, data):
        # assume that data is iterable of DataArrays
        self.data_vars = self._DataArrays_as_mapping(data)

        for key, value in self.data_vars.items():
            setattr(self, du._to_safe_name(key), value)

        if len(self.items) > 1:
            self.plot = _DatasetPlotter(self)

    def __init__(
        self,
        data: Union[Mapping[str, DataArray], Iterable[DataArray]],
        time=None,
        items=None,
        geometry: _Geometry = None,
        zn=None,
        dims=None,
    ):
        try:
            return self._init_from_DataArrays(data)
        except:
            # if not Iterable[DataArray] then let us create it...
            dataarrays = self._create_dataarrays(
                data=data, time=time, items=items, geometry=geometry, zn=zn, dims=dims
            )
            return self._init_from_DataArrays(dataarrays)

    @property
    def time(self):
        return list(self.data_vars.values())[0].time

    @time.setter
    def time(self, new_time):
        new_time = du._parse_time(new_time)
        if len(self.time) != len(new_time):
            raise ValueError("Length of new time is wrong")
        for da in self.data_vars.values():
            da.time = new_time

    @property
    def geometry(self):
        return list(self.data_vars.values())[0].geometry

    @property
    def _zn(self):
        return list(self.data_vars.values())[0]._zn

    def __repr__(self):
        if len(self) == 0:
            return "Empty <mikeio.Dataset>"

        out = ["<mikeio.Dataset>"]
        dims = [f"{self.dims[i]}:{self.shape[i]}" for i in range(self.ndim)]
        dimsstr = ", ".join(dims)
        out.append(f"Dimensions: ({dimsstr})")
        out.append(f"Time: {self.time[0]} - {self.time[-1]}")
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
        return len(self.data_vars)

    def __iter__(self):
        yield from self.data_vars.values()

    def keys(self):
        # should not be necessary
        return self.data_vars.keys()

    def __setitem__(self, key, value):

        if not isinstance(value, DataArray):
            try:
                value = DataArray(value)
                # TODO: warn that this is not the preferred way!
            except:
                raise ValueError("Use a DataArray")

        if isinstance(key, int):
            raise NotImplementedError()
            # key_str = value.name
            # new_keys = list(self.data_vars.keys())
            # new_keys.insert(key, key_str)

        self.data_vars[key] = value
        setattr(self, du._to_safe_name(key), value)

    def __getitem__(self, key) -> Union[DataArray, "Dataset"]:

        if isinstance(key, slice):
            # TODO: do we still want this behaviour?
            # slicing = slicing in time???
            # better to use sel or isel for this
            if self._slice_is_time_slice(key):
                s = self.time.slice_indexer(key.start, key.stop)
                time_steps = list(range(s.start, s.stop))
                return self.isel(time_steps, axis=0)

        key = self._key_to_str(key)
        if isinstance(key, str):
            return self.data_vars[key]

        if isinstance(key, list):
            data_vars = {}
            for v in key:
                data_vars[v] = self.data_vars[v]
            return Dataset(data_vars)

        raise TypeError(f"indexing with a {type(key)} is not (yet) supported")

    def _slice_is_time_slice(self, s):
        if (s.start is None) and (s.stop is None):
            return False
        if s.start is not None:
            if isinstance(s.start, int):
                return False
            if (
                isinstance(s.start, str)
                and (len(s.start) > 0)
                and (not s.start[0].isnumeric())
            ):
                return False
        if s.stop is not None:
            if isinstance(s.stop, int):
                return False
            if (
                isinstance(s.stop, str)
                and (len(s.stop) > 0)
                and (not s.stop[0].isnumeric())
            ):
                return False
        return True

    def _key_to_str(self, key):

        if isinstance(key, str):
            return key
        if isinstance(key, int):
            return list(self.data_vars.keys())[key]
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
        self.data_vars.__delitem__(key)
        delattr(self, du._to_safe_name(key))

    # def __getattr__(self, key) -> DataArray:
    #
    #    return self.data_vars[key]

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
            pd.DataFrame(self.data[j].flatten(), columns=[self.items[j].name]).describe(
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

        Returns
        -------
        np.ndarray
        """
        return np.stack(self.data)

    def rename(self, mapper: Mapping[str, str], inplace=False):

        if inplace:
            ds = self
        else:
            ds = self.copy()

        for key, value in mapper.items():
            da = ds.data_vars.pop(key)
            da.name = value
            ds.data_vars[value] = da

        return ds

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

        for key, value in other.data_vars.items():
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
        for j in range(ds.n_items):
            idx1 = np.where(~df12["idx1"].isna())
            newdata[j][idx1, :] = ds.data[j]
            # if there is an overlap "other" data will be used!
            idx2 = np.where(~df12["idx2"].isna())
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
        """Flip dataset updside down"""
        self.data_vars = {
            key: value.flipud() for (key, value) in self.data_vars.items()
        }
        return self

    def isel(self, idx, axis=1):
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

        axis = du._parse_axis(self.shape, self.dims, axis)
        if axis == 0:
            time = self.time[idx]
            items = self.items
            geometry = self.geometry
            zn = self._zn[idx] if self._zn else None
        else:
            time = self.time
            items = self.items
            geometry = None  # TODO
            if hasattr(self.geometry, "isel"):
                spatial_axis = du._axis_to_spatial_axis(self.dims, axis)
                geometry = self.geometry.isel(idx, axis=spatial_axis)
            zn = None  # TODO

        res = []
        for item in items:
            x = np.take(self[item.name].to_numpy(), idx, axis=axis)
            res.append(x)

        if np.isscalar(idx):
            # Selecting a single index, removes this dimension
            dims = tuple(
                [d for i, d in enumerate(self.dims) if i != axis]
            )  # TODO we will need this in many places
        else:
            dims = self.dims  # multiple points, dims is intact
        return Dataset(
            data=res, time=time, items=items, geometry=geometry, zn=zn, dims=dims
        )

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

        items = self.items
        axis = du._parse_axis(self.shape, self.dims, axis)
        time = du._time_by_agg_axis(self.time, axis)
        keepdims = du._keepdims_by_axis(axis)

        res = [
            func(self[item.name].to_numpy(), axis=axis, keepdims=keepdims, **kwargs)
            for item in items
        ]

        res = du._reshape_data_by_axis(res, self.shape, axis)

        return Dataset(res, time, items)

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

        items_in = self.items
        axis = du._parse_axis(self.shape, self.dims, axis)
        time = du._time_by_agg_axis(self.time, axis)
        keepdims = du._keepdims_by_axis(axis)

        qvec = [q] if np.isscalar(q) else q
        qtxt = [f"Quantile {q}" for q in qvec]
        itemsq = _repeat_items(items_in, qtxt)

        res = []
        for item in items_in:
            qdat = func(
                self[item.name].to_numpy(), q=q, axis=axis, keepdims=keepdims, **kwargs
            )
            for j in range(len(qvec)):
                qdat_item = qdat[j, ...] if len(qvec) > 1 else qdat
                res.append(qdat_item)

        res = du._reshape_data_by_axis(res, self.shape, axis)

        return Dataset(res, time, itemsq)

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

    def head(self, n=5):
        """Return the first n timesteps"""
        nt = len(self.time)
        n = min(n, nt)
        time_steps = range(n)
        return self.isel(time_steps, axis=0)

    def tail(self, n=5):
        """Return the last n timesteps"""
        nt = len(self.time)
        start = max(0, nt - n)
        time_steps = range(start, nt)
        return self.isel(time_steps, axis=0)

    def thin(self, step):
        """Return every n:th timesteps"""
        nt = len(self.time)
        time_steps = range(0, nt, step)
        return self.isel(time_steps, axis=0)

    def squeeze(self):
        """
        Remove axes of length 1

        Returns
        -------
        Dataset
        """

        items = self.items

        # TODO: remove this?
        if (items[0].name == "Z coordinate") and (
            items[0].type == EUMType.ItemGeometry3D
        ):
            items = deepcopy(items)
            items.pop(0)

        time = self.time

        res = [np.squeeze(self[item.name].to_numpy()) for item in items]

        ds = Dataset(res, time, items)
        return ds

    def interp_time(
        self,
        dt: Union[float, pd.DatetimeIndex, "Dataset"],
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
        if isinstance(dt, pd.DatetimeIndex):
            t_out_index = dt
        elif isinstance(dt, Dataset):
            t_out_index = dt.time
        else:
            offset = pd.tseries.offsets.DateOffset(seconds=dt)
            t_out_index = pd.date_range(
                start=self.time[0], end=self.time[-1], freq=offset
            )

        t_in = self.time.values.astype(float)
        t_out = t_out_index.values.astype(float)

        data = [
            self._interpolate_item(t_in, t_out, da, method, extrapolate, fill_value)
            for da in self
        ]

        return Dataset(data, t_out_index, self.items.copy())

    @staticmethod
    def _interpolate_item(
        intime,
        outtime,
        dataarray: DataArray,
        method: Union[str, int],
        extrapolate: bool,
        fill_value: float,
    ):
        from scipy.interpolate import interp1d

        data = dataarray.to_numpy()

        interpolator = interp1d(
            intime,
            data,
            axis=0,
            kind=method,
            bounds_error=not extrapolate,
            fill_value=fill_value,
        )
        return interpolator(outtime)

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
            data.append(dati)
        return data

    @property
    def is_equidistant(self):
        """Is Dataset equidistant in time?"""
        if len(self.time) < 3:
            return True
        return len(self.time.to_series().diff().dropna().unique()) == 1

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
    def data(self) -> Sequence[np.ndarray]:
        return [x.to_numpy() for x in self.data_vars.values()]

    @property
    def n_timesteps(self):
        """Number of time steps"""
        return len(self.time)

    @property
    def n_items(self):
        """Number of items"""
        return len(self.data_vars)

    @property
    def items(self):
        return [x.item for x in self.data_vars.values()]

    @property
    def ndim(self):
        return self[0].ndim

    @property
    def dims(self):
        return self[0].dims

    @property
    def shape(self):
        """Shape of each item"""
        return self[0].shape

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
        return self._deletevalue

    def to_dfs(self, filename):
        if self.geometry is None:
            if self.ndim == 1 and self.dims[0][0] == "t":
                self.to_dfs0(filename)
            else:
                raise ValueError("Cannot write Dataset with no geometry to file!")
        elif isinstance(self.geometry, Grid2D):
            self._to_dfs2(filename)

        elif isinstance(self.geometry, Grid1D):
            self._to_dfs1(filename)
        else:
            raise NotImplementedError(
                "Writing this type of dataset is not yet implemented"
            )

    def _to_dfs2(self, filename):
        # assumes Grid2D geometry
        from .dfs2 import write_dfs2

        write_dfs2(filename, self)

    def _to_dfs1(self, filename):
        from .dfs1 import Dfs1

        dfs = Dfs1()
        dfs.write(filename, data=self, dx=self.geometry.dx, x0=self.geometry.x0)
