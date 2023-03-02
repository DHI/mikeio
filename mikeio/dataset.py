import collections.abc
import os
import warnings
from copy import deepcopy
from datetime import datetime
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from mikecore.DfsFile import DfsSimpleType  # type: ignore

from .base import TimeSeries
from .dataarray import DataArray
from .data_utils import _to_safe_name, _get_time_idx_list, _n_selected_timesteps
from .eum import EUMType, EUMUnit, ItemInfo
from .spatial.FM_geometry import GeometryFM
from .spatial.geometry import (
    GeometryPoint2D,
    GeometryPoint3D,
    GeometryUndefined,
)
from .spatial.grid_geometry import Grid1D, Grid2D, Grid3D

from .data_plot import _DatasetPlotter


class Dataset(TimeSeries, collections.abc.MutableMapping):
    """Dataset containing one or more DataArrays with common geometry and time

    Most often obtained by reading a dfs file. But can also be
    created a sequence or dictonary of DataArrays. The mikeio.Dataset
    is inspired by and similar to the xarray.Dataset.

    The Dataset is primarily a container for one or more DataArrays
    all having the same time and geometry (and shape, dims, etc).
    For convenience, the Dataset provides access to these common properties:

    * time - a pandas.DatetimeIndex with the time instances of the data
    * geometry - a geometry object e.g. Grid2D or GeometryFM
    * shape - a tuple of array dimensions (for each DataArray)
    * dims - a tuple of dimension labels

    Selecting items
    ---------------
    Selecting a specific item "itemA" (at position 0) from a Dataset ds can be done with:

    * ds[["itemA"]] - returns a new Dataset with "itemA"
    * ds["itemA"] - returns the "itemA" DataArray
    * ds[[0]] - returns a new Dataset with "itemA"
    * ds[0] - returns the "itemA" DataArray
    * ds.itemA - returns the "itemA" DataArray

    Examples
    --------
    >>> mikeio.read("europe_wind_long_lat.dfs2")
    <mikeio.Dataset>
    dims: (time:1, y:101, x:221)
    time: 2012-01-01 00:00:00 (time-invariant)
    geometry: Grid2D (ny=101, nx=221)
    items:
    0:  Mean Sea Level Pressure <Air Pressure> (hectopascal)
    1:  Wind x-comp (10m) <Wind Velocity> (meter per sec)
    2:  Wind y-comp (10m) <Wind Velocity> (meter per sec)

    >>> mikeio.Dataset([da1, da2])
    """

    def __init__(
        self,
        data: Union[Mapping[str, DataArray], Iterable[DataArray], Sequence[np.ndarray]],
        time=None,
        items=None,
        geometry=None,
        zn=None,
        dims=None,
        validate=True,
    ):
        if not self._is_DataArrays(data):
            data = self._create_dataarrays(
                data=data, time=time, items=items, geometry=geometry, zn=zn, dims=dims
            )  # type: ignore
        self._init_from_DataArrays(data, validate=validate)

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
        data,
        time=None,
        items=None,
        geometry=None,
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
            for i in range(1, len(self)):
                da = self[i]
                first._is_compatible(da, raise_error=True)

        self._check_all_different_ids(self._data_vars.values())

        self.__itemattr = []
        for key, value in self._data_vars.items():
            self._set_name_attr(key, value)

        self.plot = _DatasetPlotter(self)

        if len(self) > 0:
            self._set_spectral_attributes(self.geometry)

        # since Dataset is MutableMapping it has values and keys by default
        # but we delete those to avoid confusion
        # self.values = None
        self.keys = None

    @property
    def values(self):
        raise AttributeError(
            "Dataset has no property 'values' - use to_numpy() instead or maybe you were looking for DataArray.values?"
        )

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
            data = Dataset._validate_item_names_and_keys(
                data
            )  # TODO is this necessary?
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
                da.name = key
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
        ids = np.zeros(len(das), dtype=np.int64)
        ids_val = np.zeros(len(das), dtype=np.int64)
        for j, da in enumerate(das):
            ids[j] = id(da)
            ids_val[j] = id(da.values)

        if len(ids) != len(np.unique(ids)):
            # DataArrays not unique! - find first duplicate and report error
            das = list(das)
            u, c = np.unique(ids, return_counts=True)
            dups = u[c > 1]
            for dup in dups:
                jj = np.where(ids == dup)[0]
                Dataset._id_of_DataArrays_equal(das[jj[0]], das[jj[1]])
        if len(ids_val) != len(np.unique(ids_val)):
            # DataArray *values* not unique! - find first duplicate and report error
            das = list(das)
            u, c = np.unique(ids_val, return_counts=True)
            dups = u[c > 1]
            for dup in dups:
                jj = np.where(ids_val == dup)[0]
                Dataset._id_of_DataArrays_equal(das[jj[0]], das[jj[1]])

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

    def _set_spectral_attributes(self, geometry):
        if hasattr(geometry, "frequencies") and hasattr(geometry, "directions"):
            self.frequencies = geometry.frequencies
            self.n_frequencies = geometry.n_frequencies
            self.directions = geometry.directions
            self.n_directions = geometry.n_directions

    # ============ end of init =============

    # ============= Basic properties/methods ===========

    @property
    def time(self) -> pd.DatetimeIndex:
        """Time axis"""
        return list(self)[0].time

    @time.setter
    def time(self, new_time):
        for da in self:
            da.time = new_time

    @property
    def start_time(self):
        """First time instance (as datetime)"""
        # TODO: use pd.Timestamp instead
        return self.time[0].to_pydatetime()

    @property
    def end_time(self):
        """Last time instance (as datetime)"""
        # TODO: use pd.Timestamp instead
        return self.time[-1].to_pydatetime()

    @property
    def timestep(self) -> Optional[float]:
        """Time step in seconds if equidistant (and at
        least two time instances); otherwise None
        """
        dt = None
        if len(self.time) > 1 and self.is_equidistant:
            dt = (self.time[1] - self.time[0]).total_seconds()  # type: ignore
        return dt

    @property
    def is_equidistant(self) -> bool:
        """Is Dataset equidistant in time?"""
        if len(self.time) < 3:
            return True
        return len(self.time.to_series().diff().dropna().unique()) == 1

    def to_numpy(self) -> np.ndarray:
        """Stack data to a single ndarray with shape (n_items, n_timesteps, ...)

        Returns
        -------
        np.ndarray
        """
        return np.stack([x.to_numpy() for x in self])

    @property
    def n_timesteps(self) -> int:
        """Number of time steps"""
        return len(self.time)

    @property
    def items(self):
        """ItemInfo for each of the DataArrays as a list"""
        return [x.item for x in self]

    @property
    def n_items(self) -> int:
        """Number of items/DataArrays, equivalent to len()"""
        return len(self._data_vars)

    @property
    def names(self):
        """Name of each of the DataArrays as a list"""
        return [da.name for da in self]

    def _ipython_key_completions_(self):
        return [x.name for x in self.items]

    @property
    def ndim(self) -> int:
        """Number of array dimensions of each DataArray"""
        return self[0].ndim

    @property
    def dims(self):
        """Named array dimensions of each DataArray"""
        return self[0].dims

    @property
    def shape(self):
        """Shape of each DataArray"""
        return self[0].shape

    @property
    def deletevalue(self):
        """File delete value"""
        return self[0].deletevalue

    @property
    def geometry(self):
        """Geometry of each DataArray"""
        return self[0].geometry

    @property
    def _zn(self) -> np.ndarray:
        return self[0]._zn

    # TODO: remove this
    @property
    def n_elements(self) -> int:
        """Number of spatial elements/points"""
        n_elem = np.prod(self.shape)
        if self.n_timesteps > 1:
            n_elem = int(n_elem / self.n_timesteps)
        return n_elem

    def describe(self, **kwargs) -> pd.DataFrame:
        """Generate descriptive statistics by wrapping :py:meth:`pandas.DataFrame.describe`"""
        data = {x.name: x.to_numpy().ravel() for x in self}
        df = pd.DataFrame(data).describe(**kwargs)

        return df

    def copy(self) -> "Dataset":
        """Returns a copy of this dataset."""

        return deepcopy(self)

    def dropna(self) -> "Dataset":
        """Remove time steps where all items are NaN"""
        if not self[0]._has_time_axis:  # type: ignore
            raise ValueError("Not available if no time axis!")

        all_index: List[int] = []
        for i in range(self.n_items):
            x = self[i].to_numpy()

            # this seems overly complicated...
            axes = tuple(range(1, x.ndim))
            idx = list(np.where(~np.isnan(x).all(axis=axes))[0])
            if i == 0:
                all_index = idx
            else:
                all_index = list(np.intersect1d(all_index, idx))

        return self.isel(all_index, axis=0)

    def flipud(self) -> "Dataset":
        """Flip data upside down (on first non-time axis)"""
        self._data_vars = {
            key: value.flipud() for (key, value) in self._data_vars.items()
        }
        return self

    def squeeze(self) -> "Dataset":
        """Remove axes of length 1

        Returns
        -------
        Dataset
        """
        res = {name: da.squeeze() for name, da in self._data_vars.items()}

        return Dataset(data=res, validate=False)

    def create_data_array(self, data, item=None) -> DataArray:
        """Create a new  DataArray with the same time and geometry as the dataset

        Examples
        --------

        >>> ds = mikeio.read("file.dfsu")
        >>> values = np.zeros(ds.Temperature.shape)
        >>> da = ds.create_data_array(values)
        >>> da_name = ds.create_data_array(values,"Foo")
        >>> da_eum = ds.create_data_array(values, item=mikeio.ItemInfo("TS", mikeio.EUMType.Temperature))
        """
        return DataArray(
            data=data, time=self.time, geometry=self.geometry, zn=self._zn, item=item
        )

    # TODO: delete this?
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

    # ============= Dataset is (almost) a MutableMapping ===========

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
                # Assignment to a key is enough indication that the user wants to name the item like this
                value.name = key
            if not is_replacement:
                self._check_already_present(value)
            self._data_vars[key] = value
            self._set_name_attr(key, value)

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

    def rename(self, mapper: Mapping[str, str], inplace=False):
        """Rename items (DataArrays) in Dataset

        Parameters
        ----------
        mapper : Mapping[str, str]
            dictionary (or similar) mapping from old to new names
        inplace : bool, optional
            Should the renaming be done in the original dataset(=True)
            or return a new(=False)?, by default False

        Returns
        -------
        Dataset

        Examples
        --------
        >>> ds = mikeio.read("tide1.dfs1")
        >>> newds = ds.rename({"Level":"Surface Elevation"})
        >>> ds.rename({"Level":"Surface Elevation"}, inplace=True)
        """
        if inplace:
            ds = self
        else:
            ds = self.copy()

        for old_name, new_name in mapper.items():
            da = ds._data_vars.pop(old_name)
            da.name = new_name
            ds._data_vars[new_name] = da
            ds._del_name_attr(old_name)
            ds._set_name_attr(new_name, da)

        return ds

    def _set_name_attr(self, name: str, value: DataArray):
        name = _to_safe_name(name)
        if name not in self.__itemattr:
            self.__itemattr.append(name)  # keep track of what we insert
        setattr(self, name, value)

    def _del_name_attr(self, name: str):
        name = _to_safe_name(name)
        if name in self.__itemattr:
            self.__itemattr.remove(name)
            delattr(self, name)

    def __getitem__(self, key) -> Union[DataArray, "Dataset"]:

        # select time steps
        if (
            isinstance(key, Sequence) and not isinstance(key, str)
        ) and self._is_key_time(key[0]):
            key = pd.DatetimeIndex(key)  # type: ignore
        if isinstance(key, pd.DatetimeIndex) or self._is_key_time(key):
            time_steps = _get_time_idx_list(self.time, key)
            if _n_selected_timesteps(self.time, time_steps) == 0:
                raise IndexError("No timesteps found!")
            return self.isel(time_steps, axis=0)
        if isinstance(key, slice):
            if self._is_slice_time_slice(key):
                try:
                    s = self.time.slice_indexer(key.start, key.stop)
                    time_steps = list(range(s.start, s.stop))
                except:
                    time_steps = list(range(*key.indices(len(self.time))))
                return self.isel(time_steps, axis=0)

        if self._multi_indexing_attempted(key):
            raise TypeError(
                f"Indexing with key {key} failed. Dataset does not allow multi-indexing. Use isel() or sel() instead."
            )

        # select items
        key = self._key_to_str(key)

        if isinstance(key, str):
            if key in self._data_vars.keys():
                return self._data_vars[key]

            if "*" in key:
                import fnmatch

                data_vars = {
                    k: da
                    for k, da in self._data_vars.items()
                    if fnmatch.fnmatch(k, key)
                }
                return Dataset(data=data_vars, validate=False)
            else:
                item_names = ",".join(self._data_vars.keys())
                raise KeyError(f"No item named: {key}. Valid items: {item_names}")

        if isinstance(key, Iterable):
            data_vars = {}
            for v in key:
                data_vars[v] = self._data_vars[v]
            return Dataset(data=data_vars, validate=False)

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

    def _multi_indexing_attempted(self, key) -> bool:
        # find out if user is attempting ds[2, :, 1] or similar (not allowed)
        # this is not bullet-proof, but a good estimate
        if not isinstance(key, tuple):
            return False
        for k in key:
            if isinstance(k, slice):
                # warnings.warn(f"Key is a tuple containing a slice")
                return True
            if not isinstance(k, (str, int)):
                # warnings.warn(f"Key is a tuple containing illegal type {type(k)}")
                return True
        if len(set(key)) != len(key):
            return True
        warnings.warn(
            f"A tuple of item numbers/names was provided as index to Dataset. This can lead to ambiguity and it is recommended to use a list instead."
        )
        return False

    # TODO change this to return a single type
    def _key_to_str(self, key: Union[str, int, slice, Iterable[str], Iterable[int]]):
        """Translate item selection key to str (or List[str])"""
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

    # ============ select/interp =============

    def isel(self, idx=None, axis=0, **kwargs):
        """Return a new Dataset whose data is given by
        integer indexing along the specified dimension(s).

        The spatial parameters available depend on the dims
        (i.e. geometry) of the Dataset:

        * Grid1D: x
        * Grid2D: x, y
        * Grid3D: x, y, z
        * GeometryFM: element

        Parameters
        ----------
        idx: int, scalar or array_like
        axis: (int, str, None), optional
            axis number or "time", by default 0
        time : int, optional
            time index,by default None
        x : int, optional
            x index, by default None
        y : int, optional
            y index, by default None
        z : int, optional
            z index, by default None
        element : int, optional
            Bounding box of coordinates (left lower and right upper)
            to be selected, by default None

        Returns
        -------
        Dataset
            dataset with subset

        Examples
        --------
        >>> ds = mikeio.read("europe_wind_long_lat.dfs2")
        >>> ds.isel(time=-1)
        >>> ds.isel(x=slice(10,20), y=slice(40,60))
        >>> ds.isel(y=34)

        >>> ds = mikeio.read("tests/testdata/HD2D.dfsu")
        >>> ds2 = ds.isel(time=[0,1,2])
        >>> ds3 = ds2.isel(elements=[100,200])
        """
        res = [da.isel(idx=idx, axis=axis, **kwargs) for da in self]
        return Dataset(data=res, validate=False)

    def sel(
        self,
        **kwargs,
    ) -> "Dataset":
        """Return a new Dataset whose data is given by
        selecting index labels along the specified dimension(s).

        In contrast to Dataset.isel, indexers for this method
        should use labels instead of integers.

        The spatial parameters available depend on the geometry of the Dataset:

        * Grid1D: x
        * Grid2D: x, y, coords, area
        * Grid3D: [not yet implemented! use isel instead]
        * GeometryFM: (x,y), coords, area
        * GeometryFMLayered: (x,y,z), coords, area, layers

        Parameters
        ----------
        time : Union[str, pd.DatetimeIndex, Dataset], optional
            time labels e.g. "2018-01" or slice("2018-1-1","2019-1-1"),
            by default None
        x : float, optional
            x-coordinate of point to be selected, by default None
        y : float, optional
            y-coordinate of point to be selected, by default None
        z : float, optional
            z-coordinate of point to be selected, by default None
        coords : np.array(float,float), optional
            As an alternative to specifying x, y and z individually,
            the argument coords can be used instead.
            (x,y)- or (x,y,z)-coordinates of point to be selected,
            by default None
        area : (float, float, float, float), optional
            Bounding box of coordinates (left lower and right upper)
            to be selected, by default None
        layers : int or str or list, optional
            layer(s) to be selected: "top", "bottom" or layer number
            from bottom 0,1,2,... or from the top -1,-2,... or as
            list of these; only for layered dfsu, by default None

        Returns
        -------
        Dataset
            new Dataset with selected data

        See Also
        --------
        isel : Select data using integer indexing

        Examples
        --------
        >>> ds = mikeio.read("random.dfs1")
        >>> ds.sel(time=slice(None, "2012-1-1 00:02"))
        >>> ds.sel(x=100)

        >>> ds = mikeio.read("oresund_sigma_z.dfsu")
        >>> ds.sel(time="1997-09-15")
        >>> ds.sel(x=340000, y=6160000, z=-3)
        >>> ds.sel(area=(340000, 6160000, 350000, 6170000))
        >>> ds.sel(layers="bottom")
        """
        res = [da.sel(**kwargs) for da in self]
        return Dataset(data=res, validate=False)

    def interp(
        self,
        *,
        time: Optional[Union[pd.DatetimeIndex, "DataArray"]] = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        n_nearest: int = 3,
        **kwargs,
    ) -> "Dataset":
        """Interpolate data in time and space

        This method currently has limited functionality for
        spatial interpolation. It will be extended in the future.

        The spatial parameters available depend on the geometry of the Dataset:

        * Grid1D: x
        * Grid2D: x, y
        * Grid3D: [not yet implemented!]
        * GeometryFM: (x,y)
        * GeometryFMLayered: (x,y) [surface point will be returned!]

        Parameters
        ----------
        time : Union[float, pd.DatetimeIndex, Dataset], optional
            timestep in seconds or discrete time instances given by
            pd.DatetimeIndex (typically from another Dataset
            da2.time), by default None (=don't interp in time)
        x : float, optional
            x-coordinate of point to be interpolated to, by default None
        y : float, optional
            y-coordinate of point to be interpolated to, by default None
        n_nearest : int, optional
            When using IDW interpolation, how many nearest points should
            be used, by default: 3

        Returns
        -------
        Dataset
            new Dataset with interped data

        See Also
        --------
        sel : Select data using label indexing
        interp_like : Interp to another time/space of another DataSet
        interp_time : Interp in the time direction only

        Examples
        --------
        >>> ds = mikeio.read("random.dfs1")
        >>> ds.interp(time=3600)
        >>> ds.interp(x=110)

        >>> ds = mikeio.read("HD2D.dfsu")
        >>> ds.interp(x=340000, y=6160000)
        """
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
            ds = Dataset(das, validate=False)
        else:
            ds = Dataset([da for da in self], validate=False)

        # interp in time
        if isinstance(time, (pd.DatetimeIndex, DataArray)):
            ds = ds.interp_time(time)

        return ds

    def __dataset_read_item_time_func(
        self, item: int, step: int
    ) -> Tuple[np.ndarray, float]:
        "Used by _extract_track"

        data = self[item].isel(time=step).to_numpy()
        time = (self.time[step] - self.time[0]).total_seconds()  # type: ignore

        return data, time

    def extract_track(self, track, method="nearest", dtype=np.float32):
        """
        Extract data along a moving track

        Parameters
        ---------
        track: pandas.DataFrame
            with DatetimeIndex and (x, y) of track points as first two columns
            x,y coordinates must be in same coordinate system as dfsu
        track: str
            filename of csv or dfs0 file containing t,x,y
        method: str, optional
            Spatial interpolation method ('nearest' or 'inverse_distance')
            default='nearest'

        Returns
        -------
        Dataset
            A dataset with data dimension t
            The first two items will be x- and y- coordinates of track
        """
        from .track import _extract_track

        item_numbers = list(range(self.n_items))
        time_steps = list(range(self.n_timesteps))

        return _extract_track(
            deletevalue=self.deletevalue,
            start_time=self.start_time,
            end_time=self.end_time,
            timestep=self.timestep,
            geometry=self.geometry,
            n_elements=self.shape[1],  # TODO is there a better way to find out this?
            track=track,
            items=self.items,
            time_steps=time_steps,
            item_numbers=item_numbers,
            method=method,
            dtype=dtype,
            data_read_func=lambda item, step: self.__dataset_read_item_time_func(
                item, step
            ),
        )

    def interp_time(
        self,
        dt: Optional[Union[float, pd.DatetimeIndex, "Dataset", DataArray]] = None,
        *,
        freq: Optional[str] = None,
        method="linear",
        extrapolate=True,
        fill_value=np.nan,
    ) -> "Dataset":
        """Temporal interpolation

        Wrapper of :py:class:`scipy.interpolate.interp1d`

        Parameters
        ----------
        dt: float or pd.DatetimeIndex or Dataset
            output timestep in seconds or discrete time instances given
            as a pd.DatetimeIndex (typically from another Dataset
            ds2.time)
        freq: str
            pandas frequency
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
        >>> dsi = ds.interp_time(freq='2H')
        """
        if freq:
            dt = pd.to_timedelta(freq).total_seconds()
        else:
            if dt is None:
                raise ValueError("You must specify either dt or freq")

        das = [
            da.interp_time(
                dt=dt, method=method, extrapolate=extrapolate, fill_value=fill_value
            )
            for da in self
        ]

        return Dataset(das)

    def interp_na(self, axis="time", **kwargs) -> "Dataset":
        ds = self.copy()
        for da in ds:
            da.values = da.interp_na(axis=axis, **kwargs).values

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
        if not (isinstance(self.geometry, GeometryFM) and self.geometry.is_2d):
            raise NotImplementedError(
                "Currently only supports interpolating from 2d flexible mesh data!"
            )

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
        ds = Dataset(das, validate=False)

        if hasattr(other, "time"):
            ds = ds.interp_time(other.time)

        return ds

    # ============= Combine/concat ===========

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

    @staticmethod
    def concat(datasets: Sequence["Dataset"], keep="last") -> "Dataset":
        """Concatenate Datasets along the time axis

        Parameters
        ---------
        datasets: sequence of Datasets
        keep: str, optional
            TODO Yet to be implemented, default: last

        Returns
        -------
        Dataset
            concatenated dataset

        Examples
        --------
        >>> import mikeio
        >>> ds1 = mikeio.read("HD2D.dfsu", time=[0,1])
        >>> ds2 = mikeio.read("HD2D.dfsu", time=[2,3])
        >>> ds1.n_timesteps
        2
        >>> ds3 = Dataset.concat([ds1,ds2])
        >>> ds3.n_timesteps
        4
        """

        if keep != "last":
            raise NotImplementedError(
                "Last values is the only available option at the moment."
            )
        ds = datasets[0].copy()
        for dsj in datasets[1:]:
            ds = ds._concat_time(dsj, copy=False)

        return ds

    @staticmethod
    def merge(datasets: Sequence["Dataset"]) -> "Dataset":
        """Merge Datasets along the item dimension

        Parameters
        ---------
        datasets: sequence of Datasets

        Returns
        -------
        Dataset
            merged dataset
        """
        ds = datasets[0].copy()
        for dsj in datasets[1:]:
            ds = ds._append_items(dsj, copy=False)

        return ds

    def _concat_time(self, other, copy=True) -> "Dataset":
        self._check_all_items_match(other)
        # assuming time is always first dimension we can skip / keep it by bool
        start_dim = int("time" in self.dims)
        if not np.all(
            self.shape[start_dim:] == other.shape[int("time" in other.dims) :]
        ):
            # if not np.all(self.shape[1:] == other.shape[1:]):
            raise ValueError("Shape of the datasets must match (except time dimension)")
        if hasattr(self, "time"):  # using attribute instead of dim checking. Works
            ds = self.copy() if copy else self
        else:
            raise ValueError(
                "Datasets cannot be concatenated as they have no time attribute!"
            )

        s1 = pd.Series(np.arange(len(ds.time)), index=ds.time, name="idx1")
        s2 = pd.Series(np.arange(len(other.time)), index=other.time, name="idx2")
        df12 = pd.concat([s1, s2], axis=1)

        newtime = df12.index
        newdata = self.create_empty_data(
            n_items=ds.n_items, n_timesteps=len(newtime), shape=ds.shape[start_dim:]
        )
        idx1 = np.where(~df12["idx1"].isna())
        idx2 = np.where(~df12["idx2"].isna())
        for j in range(ds.n_items):
            #    # if there is an overlap "other" data will be used!
            newdata[j][idx1] = ds[j].to_numpy()
            newdata[j][idx2] = other[j].to_numpy()

        zn = None
        if self._zn is not None:
            zshape = (len(newtime), self._zn.shape[start_dim])
            zn = np.zeros(shape=zshape, dtype=self._zn.dtype)
            zn[idx1, :] = self._zn
            zn[idx2, :] = other._zn

        return Dataset(
            newdata, time=newtime, items=ds.items, geometry=ds.geometry, zn=zn
        )

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

    # ============ aggregate =============

    def aggregate(
        self, axis=0, func=np.nanmean, **kwargs
    ) -> Union["Dataset", "DataArray"]:
        """Aggregate along an axis

        Parameters
        ----------
        axis: (int, str, None), optional
            axis number or "time", "space" or "items", by default 0
        func: function, optional
            default np.nanmean

        Returns
        -------
        Dataset
            dataset with aggregated values
        """
        if axis == "items":
            if self.n_items <= 1:
                return self

            if "keepdims" in kwargs:
                warnings.warn(
                    "The keepdims arguments is deprecated. The result will always be a Dataset.",
                    FutureWarning,
                )

            keepdims = kwargs.pop("keepdims", False)
            name = kwargs.pop("name", func.__name__)
            data = func(self.to_numpy(), axis=0, keepdims=False, **kwargs)
            item = self._agg_item_from_items(self.items, name)
            da = DataArray(
                data=data,
                time=self.time,
                item=item,
                geometry=self.geometry,
                dims=self.dims,
                zn=self._zn,
            )
            if not keepdims:
                warnings.warn(
                    "The keepdims arguments is deprecated. The result will always be a Dataset.",
                    FutureWarning,
                )
            return Dataset([da], validate=False) if keepdims else da
        else:
            res = {
                name: da.aggregate(axis=axis, func=func, **kwargs)
                for name, da in self._data_vars.items()
            }
            return Dataset(data=res, validate=False)

    @staticmethod
    def _agg_item_from_items(items, name):
        it_type = (
            items[0].type
            if all([it.type == items[0].type for it in items])
            else EUMType.Undefined
        )
        it_unit = (
            items[0].unit
            if all([it.unit == items[0].unit for it in items])
            else EUMUnit.undefined
        )
        return ItemInfo(name, it_type, it_unit)

    def quantile(self, q, *, axis=0, **kwargs) -> Union["Dataset", "DataArray"]:
        """Compute the q-th quantile of the data along the specified axis.

        Wrapping np.quantile

        Parameters
        ----------
        q: array_like of float
            Quantile or sequence of quantiles to compute,
            which must be between 0 and 1 inclusive.
        axis: (int, str, None), optional
            axis number or "time", "space" or "items", by default 0

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

    def nanquantile(self, q, *, axis=0, **kwargs) -> Union["Dataset", "DataArray"]:
        """Compute the q-th quantile of the data along the specified axis, while ignoring nan values.

        Wrapping np.nanquantile

        Parameters
        ----------
        q: array_like of float
            Quantile or sequence of quantiles to compute,
            which must be between 0 and 1 inclusive.
        axis: (int, str, None), optional
            axis number or "time", "space" or "items", by default 0

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

    def _quantile(
        self, q, *, axis=0, func=np.quantile, **kwargs
    ) -> Union["Dataset", "DataArray"]:

        if axis == "items":
            keepdims = kwargs.pop("keepdims", False)
            if self.n_items <= 1:
                return self  # or raise ValueError?
            if np.isscalar(q):
                data = func(self.to_numpy(), q=q, axis=0, keepdims=False, **kwargs)
                item = self._agg_item_from_items(self.items, f"Quantile {str(q)}")
                da = DataArray(
                    data=data,
                    time=self.time,
                    item=item,
                    geometry=self.geometry,
                    dims=self.dims,
                    zn=self._zn,
                )
                return Dataset([da], validate=False) if keepdims else da
            else:
                if keepdims:
                    raise ValueError("Cannot keepdims for multiple quantiles")
                res = []
                for quantile in q:
                    qd = self._quantile(q=quantile, axis=axis, func=func, **kwargs)
                    assert isinstance(qd, DataArray)
                    res.append(qd)
                return Dataset(data=res, validate=False)
        else:
            if np.isscalar(q):
                res = [da._quantile(q=q, axis=axis, func=func) for da in self]
            else:
                res = []

                for name, da in self._data_vars.items():
                    for quantile in q:
                        qd = da._quantile(q=quantile, axis=axis, func=func)
                        assert isinstance(qd, DataArray)
                        newname = f"Quantile {quantile}, {name}"
                        qd.name = newname
                        res.append(qd)

            return Dataset(data=res, validate=False)

    def max(self, axis=0, **kwargs) -> Union["Dataset", "DataArray"]:
        """Max value along an axis

        Parameters
        ----------
        axis: (int, str, None), optional
            axis number or "time", "space" or "items", by default 0

        Returns
        -------
        Dataset
            dataset with max values

        See Also
        --------
            nanmax : Max values with NaN values removed
        """
        return self.aggregate(axis=axis, func=np.max, **kwargs)

    def min(self, axis=0, **kwargs) -> Union["Dataset", "DataArray"]:
        """Min value along an axis

        Parameters
        ----------
        axis: (int, str, None), optional
            axis number or "time", "space" or "items", by default 0

        Returns
        -------
        Dataset
            dataset with min values

        See Also
        --------
            nanmin : Min values with NaN values removed
        """
        return self.aggregate(axis=axis, func=np.min, **kwargs)

    def mean(self, axis=0, **kwargs) -> Union["Dataset", "DataArray"]:
        """Mean value along an axis

        Parameters
        ----------
        axis: (int, str, None), optional
            axis number or "time", "space" or "items", by default 0

        Returns
        -------
        Dataset
            dataset with mean values

        See Also
        --------
            nanmean : Mean values with NaN values removed
            average : Weighted average
        """
        return self.aggregate(axis=axis, func=np.mean, **kwargs)

    def std(self, axis=0, **kwargs) -> Union["Dataset", "DataArray"]:
        """Standard deviation along an axis

        Parameters
        ----------
        axis: (int, str, None), optional
            axis number or "time", "space" or "items", by default 0

        Returns
        -------
        Dataset
            dataset with standard deviation values

        See Also
        --------
            nanstd : Standard deviation with NaN values removed
        """
        return self.aggregate(axis=axis, func=np.std, **kwargs)

    def ptp(self, axis=0, **kwargs) -> Union["Dataset", "DataArray"]:
        """Range (max - min) a.k.a Peak to Peak along an axis
        Parameters
        ----------
        axis: (int, str, None), optional
            axis number or "time", "space" or "items", by default 0

        Returns
        -------
        Dataset
            dataset with peak to peak values
        """
        return self.aggregate(axis=axis, func=np.ptp, **kwargs)

    def average(self, weights, axis=0, **kwargs) -> Union["Dataset", "DataArray"]:
        """Compute the weighted average along the specified axis.

        Parameters
        ----------
        axis: (int, str, None), optional
            axis number or "time", "space" or "items", by default 0

        Returns
        -------
        Dataset
            dataset with weighted average values

        See Also
        --------
            nanmean : Mean values with NaN values removed
            aggregate : Weighted average

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

        return self.aggregate(axis=axis, func=func, **kwargs)

    def nanmax(self, axis=0, **kwargs) -> Union["Dataset", "DataArray"]:
        """Max value along an axis (NaN removed)

        Parameters
        ----------
        axis: (int, str, None), optional
            axis number or "time", "space" or "items", by default 0

        See Also
        --------
            max : Mean values

        Returns
        -------
        Dataset
            dataset with max values
        """
        return self.aggregate(axis=axis, func=np.nanmax, **kwargs)

    def nanmin(self, axis=0, **kwargs) -> Union["Dataset", "DataArray"]:
        """Min value along an axis (NaN removed)

        Parameters
        ----------
        axis: (int, str, None), optional
            axis number or "time", "space" or "items", by default 0

        Returns
        -------
        Dataset
            dataset with min values
        """
        return self.aggregate(axis=axis, func=np.nanmin, **kwargs)

    def nanmean(self, axis=0, **kwargs) -> Union["Dataset", "DataArray"]:
        """Mean value along an axis (NaN removed)

        Parameters
        ----------
        axis: (int, str, None), optional
            axis number or "time", "space" or "items", by default 0

        Returns
        -------
        Dataset
            dataset with mean values
        """
        return self.aggregate(axis=axis, func=np.nanmean, **kwargs)

    def nanstd(self, axis=0, **kwargs) -> Union["Dataset", "DataArray"]:
        """Standard deviation along an axis (NaN removed)

        Parameters
        ----------
        axis: (int, str, None), optional
            axis number or "time", "space" or "items", by default 0

        Returns
        -------
        Dataset
            dataset with standard deviation values

        See Also
        --------
            std : Standard deviation
        """
        return self.aggregate(axis=axis, func=np.nanstd, **kwargs)

    # ============ arithmetic/Math =============

    def __radd__(self, other) -> "Dataset":
        return self.__add__(other)

    def __add__(self, other) -> "Dataset":
        if isinstance(other, self.__class__):
            return self._add_dataset(other)
        else:
            return self._add_value(other)

    def __rsub__(self, other) -> "Dataset":
        ds = self.__mul__(-1.0)
        return other + ds

    def __sub__(self, other) -> "Dataset":
        if isinstance(other, self.__class__):
            return self._add_dataset(other, sign=-1.0)
        else:
            return self._add_value(-other)

    def __rmul__(self, other) -> "Dataset":
        return self.__mul__(other)

    def __mul__(self, other) -> "Dataset":
        if isinstance(other, self.__class__):
            raise ValueError("Multiplication is not possible for two Datasets")
        else:
            return self._multiply_value(other)

    def _add_dataset(self, other, sign=1.0) -> "Dataset":
        self._check_datasets_match(other)
        try:
            data = [
                self[x].to_numpy() + sign * other[y].to_numpy()
                for x, y in zip(self.items, other.items)
            ]
        except:
            raise ValueError("Could not add data in Dataset")
        newds = self.copy()
        for j in range(len(self)):
            newds[j].values = data[j]  # type: ignore
        return newds

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

    def _add_value(self, value) -> "Dataset":
        try:
            data = [value + self[x].to_numpy() for x in self.items]
        except:
            raise ValueError(f"{value} could not be added to Dataset")
        items = deepcopy(self.items)
        time = self.time.copy()
        return Dataset(
            data,
            time=time,
            items=items,
            geometry=self.geometry,
            zn=self._zn,
            validate=False,
        )

    def _multiply_value(self, value) -> "Dataset":
        try:
            data = [value * self[x].to_numpy() for x in self.items]
        except:
            raise ValueError(f"{value} could not be multiplied to Dataset")
        items = deepcopy(self.items)
        time = self.time.copy()
        return Dataset(
            data,
            time=time,
            items=items,
            geometry=self.geometry,
            zn=self._zn,
            validate=False,
        )

    # ===============================================

    def to_dataframe(
        self, *, unit_in_name: bool = False, round_time: Union[str, bool] = "ms"
    ) -> pd.DataFrame:
        """Convert Dataset to a Pandas DataFrame

        Parameters
        ----------
        unit_in_name: bool, optional
            include unit in column name, default False,
        round_time: str, bool, optional
            round time to, by default "ms", use False to avoid rounding

        Returns
        -------
        pd.DataFrame
        """
        if self.ndim > 1:
            raise ValueError(
                "Only data with a single dimension can be converted to a dataframe. Hint: use `squeeze` to remove singleton dimensions or `isel` to create a subset."
            )

        if unit_in_name:
            data = {
                f"{item.name} ({item.unit.name})": item.to_numpy().ravel()
                for item in self
            }
        else:
            data = {item.name: item.to_numpy().ravel() for item in self}
        df = pd.DataFrame(data, index=self.time)

        if round_time:
            rounded_idx = pd.DatetimeIndex(self.time).round(round_time)  # type: ignore
            df.index = pd.DatetimeIndex(rounded_idx, freq="infer")
        else:
            df.index = pd.DatetimeIndex(self.time, freq="infer")

        return df

    def to_dfs(self, filename, **kwargs):
        """Write dataset to a new dfs file

        Parameters
        ----------
        filename: str
            full path to the new dfs file
        dtype: str, np.dtype, DfsSimpleType, optional
            Dfs0 only: set the dfs data type of the written data
            to e.g. np.float64, by default: DfsSimpleType.Float (=np.float32)
        """

        filename = str(filename)

        if isinstance(
            self.geometry, (GeometryPoint2D, GeometryPoint3D, GeometryUndefined)
        ):

            if self.ndim == 0:  # Not very common, but still...
                self._validate_extension(filename, ".dfs0")
                self._to_dfs0(filename, **kwargs)
            elif self.ndim == 1 and self[0]._has_time_axis:
                self._validate_extension(filename, ".dfs0")
                self._to_dfs0(filename, **kwargs)
            else:
                raise ValueError("Cannot write Dataset with no geometry to file!")
        elif isinstance(self.geometry, Grid2D):
            self._validate_extension(filename, ".dfs2")
            self._to_dfs2(filename)
        elif isinstance(self.geometry, Grid3D):
            self._validate_extension(filename, ".dfs3")
            self._to_dfs3(filename)

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

    @staticmethod
    def _validate_extension(filename, valid_extension):
        ext = os.path.splitext(filename)[1].lower()
        if ext != valid_extension:
            raise ValueError(f"File extension must be {valid_extension}")

    def _to_dfs0(self, filename, **kwargs):
        from .dfs0 import _write_dfs0

        dtype = kwargs.get("dtype", DfsSimpleType.Float)

        _write_dfs0(filename, self, dtype=dtype)

    def _to_dfs2(self, filename):
        # assumes Grid2D geometry
        from .dfs2 import write_dfs2

        write_dfs2(filename, self)

    def _to_dfs3(self, filename):
        # assumes Grid3D geometry
        from .dfs3 import write_dfs3

        write_dfs3(filename, self)

    def _to_dfs1(self, filename):
        from .dfs1 import Dfs1

        dfs = Dfs1()
        dfs.write(filename, data=self, dx=self.geometry.dx, x0=self.geometry._x0)

    def _to_dfsu(self, filename):
        from .dfsu import _write_dfsu

        _write_dfsu(filename, self)

    def to_xarray(self):
        """Export to xarray.Dataset"""
        import xarray

        data = {da.name: da.to_xarray() for da in self}
        return xarray.Dataset(data)

    # ===============================================

    def __repr__(self) -> str:
        if len(self) == 0:
            return "Empty <mikeio.Dataset>"
        da = self[0]
        out = ["<mikeio.Dataset>", da._dims_txt(), da._time_txt(), da._geometry_txt()]  # type: ignore
        out = [x for x in out if x is not None]

        if self.n_items > 10:
            out.append(f"number of items: {self.n_items}")
        else:
            out.append("items:")
            for i, item in enumerate(self.items):
                out.append(f"  {i}:  {item}")

        return str.join("\n", out)
