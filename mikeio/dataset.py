import warnings
from typing import Sequence, Union, List
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
from copy import deepcopy
from mikeio.eum import EUMType, ItemInfo

from .base import TimeSeries


class Dataset(TimeSeries):

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

    def __init__(
        self,
        data: Union[List[np.ndarray], float],
        time: Union[pd.DatetimeIndex, str],
        items: Union[List[ItemInfo], List[EUMType], List[str]] = None,
    ):

        item_infos: List[ItemInfo] = []

        self._deletevalue = Dataset.deletevalue

        if isinstance(time, str):
            # default single-step time
            time = self.create_time(time)

        if np.isscalar(data) and isinstance(items, Sequence):
            # create empty dataset
            n_elements = data
            n_items = len(items)
            n_timesteps = len(time)
            data = self.create_empty_data(
                n_items=n_items, n_timesteps=n_timesteps, n_elements=n_elements
            )

        if isinstance(data, Sequence):
            n_items = len(data)
            n_timesteps = data[0].shape[0]

        if items is None:
            # default Undefined items
            for j in range(n_items):
                item_infos.append(ItemInfo(f"Item {j+1}"))
        else:
            for item in items:
                if isinstance(item, EUMType) or isinstance(item, str):
                    item_infos.append(ItemInfo(item))
                elif isinstance(item, ItemInfo):
                    item_infos.append(item)
                else:
                    raise ValueError(f"items of type: {type(item)} is not supported")

            if len(items) != n_items:
                raise ValueError(
                    f"Number of items in iteminfo {len(items)} doesn't match the data {n_items}."
                )

        if len(time) != n_timesteps:
            raise ValueError(
                f"Number of timesteps in time {len(time)} doesn't match the data {n_timesteps}."
            )

        self.data = data
        self.time = pd.DatetimeIndex(time)

        self._items = item_infos

    def __repr__(self):

        out = ["<mikeio.DataSet>"]
        out.append(f"Dimensions: {self.shape}")
        out.append(f"Time: {self.time[0]} - {self.time[-1]}")
        if not self.is_equidistant:
            out.append("-- Non-equidistant time axis --")
        if self.n_items > 10:
            out.append(f"Number of items: {self.n_items}")
        else:
            out.append("Items:")
            for i, item in enumerate(self.items):
                out.append(f"  {i}:  {item}")

        return str.join("\n", out)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, x):

        if isinstance(x, slice):
            s = self.time.slice_indexer(x.start, x.stop)
            time_steps = list(range(s.start, s.stop))
            return self.isel(time_steps, axis=0)

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

        raise ValueError(f"indexing with a {type(x)} is not (yet) supported")

    def copy(self):
        "Returns a copy of this dataset."

        items = deepcopy(self.items)
        data = [self[x].copy() for x in self.items]
        time = self.time.copy()

        return Dataset(data, time, items)

    def dropna(self):
        "Remove time steps where all items are NaN"

        # TODO consider all items
        x = self[0]

        # this seems overly complicated...
        axes = tuple(range(1, x.ndim))
        idx = np.where(~np.isnan(x).all(axis=axes))
        idx = list(idx[0])

        return self.isel(idx, axis=0)

    def flipud(self):
        "Flip dataset updside down"

        self.data = [np.flip(self[x], axis=1) for x in self.items]
        return self

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

    def aggregate(self, axis=1, func=np.nanmean):
        """Aggregate along an axis


        Parameters
        ----------
        axis: int, optional
            default 1= first spatial axis
        func: function, optional
            default np.nanmean

        Returns
        -------
        Dataset
            dataset with aggregated values
        """

        items = self.items

        if items[0].name == "Z coordinate":
            items = deepcopy(items)
            items.pop(0)

        if axis == 0:
            time = pd.DatetimeIndex([self.time[0]])
            keepdims = True
        else:
            time = self.time
            keepdims = False

        res = [func(self[item.name], axis=axis, keepdims=keepdims) for item in items]

        ds = Dataset(res, time, items)
        return ds

    def max(self, axis=1):
        """Max value along an axis

        Parameters
        ----------
        axis: int, optional
            default 1= first spatial axis

        Returns
        -------
        Dataset
            dataset with max value

        See Also
        --------
            nanmax : Max values with NaN values removed
        """
        return self.aggregate(axis=axis, func=np.max)

    def min(self, axis=1):
        """Min value along an axis

        Parameters
        ----------
        axis: int, optional
            default 1= first spatial axis

        Returns
        -------
        Dataset
            dataset with max value

        See Also
        --------
            nanmin : Min values with NaN values removed
        """
        return self.aggregate(axis=axis, func=np.min)

    def mean(self, axis=1):
        """Mean value along an axis

        Parameters
        ----------
        axis: int, optional
            default 1= first spatial axis

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

    def average(self, weights, axis=1):
        """
        Compute the weighted average along the specified axis.

        Parameters
        ----------
        axis: int, optional
            default 1= first spatial axis

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
        >>> ds2 = ds.average(weights=area)
        """

        def func(x, axis, keepdims):
            if keepdims:
                raise NotImplementedError()

            return np.average(x, weights=weights, axis=axis)

        return self.aggregate(axis=axis, func=func)

    def nanmax(self, axis=1):
        """Max value along an axis (NaN removed)

        Parameters
        ----------
        axis: int, optional
            default 1= first spatial axis

        Returns
        -------
        Dataset
            dataset with max value
        """
        return self.aggregate(axis=axis, func=np.nanmax)

    def nanmin(self, axis=1):
        """Min value along an axis (NaN removed)

        Parameters
        ----------
        axis: int, optional
            default 1= first spatial axis

        Returns
        -------
        Dataset
            dataset with max value
        """
        return self.aggregate(axis=axis, func=np.nanmin)

    def nanmean(self, axis=1):
        """Mean value along an axis (NaN removed)

        Parameters
        ----------
        axis: int, optional
            default 1= first spatial axis

        Returns
        -------
        Dataset
            dataset with mean value
        """
        return self.aggregate(axis=axis, func=np.nanmean)

    def head(self, n=5):
        "Return the first n timesteps"
        nt = len(self.time)
        n = min(n, nt)
        time_steps = range(n)
        return self.isel(time_steps, axis=0)

    def tail(self, n=5):
        "Return the last n timesteps"
        nt = len(self.time)
        start = max(0, nt - n)
        time_steps = range(start, nt)
        return self.isel(time_steps, axis=0)

    def thin(self, step):
        "Return every n:th timesteps"
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

        if items[0].name == "Z coordinate":
            items = deepcopy(items)
            items.pop(0)

        time = self.time

        res = [np.squeeze(self[item.name]) for item in items]

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
            Specifies the kind of interpolation as a string (‘linear’, ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, ‘next’, where ‘zero’, ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation of zeroth, first, second or third order; ‘previous’ and ‘next’ simply return the previous or next value of the point) or as an integer specifying the order of the spline interpolator to use. Default is ‘linear’.
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
        <mikeio.DataSet>
        Dimensions: (9, 884)
        Time: 1985-08-06 07:00:00 - 1985-08-07 03:00:00
        Items:
        0:  Surface elevation <Surface Elevation> (meter)
        1:  U velocity <u velocity component> (meter per sec)
        2:  V velocity <v velocity component> (meter per sec)
        3:  Current speed <Current Speed> (meter per sec)
        >>> dsi = ds.interp_time(dt=1800)
        >>> dsi
        <mikeio.DataSet>
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
            self._interpolate_item(t_in, t_out, item, method, extrapolate, fill_value)
            for item in self
        ]

        return Dataset(data, t_out_index, self.items.copy())

    @staticmethod
    def _interpolate_item(intime, outtime, dataitem, method, extrapolate, fill_value):

        interpolator = interp1d(
            intime,
            dataitem,
            axis=0,
            kind=method,
            bounds_error=not extrapolate,
            fill_value=fill_value,
        )
        return interpolator(outtime)

    def to_dataframe(self, unit_in_name=False):
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

    @staticmethod
    def create_time(start_time=None, dt=None, n_timesteps=None, end_time=None):
        """create a equidistant time axis (calendar axis)

        Parameters
        ----------
        start_time : datetime or str, optional
            start_time, by default None (1970-1-1 00:00:00)
            can optionally contain end_time e.g. '2018-01-01, 2018-02-01'
        dt : float, optional
            time step in seconds, by default None
        n_timesteps : int, optional
            number of timesteps, by default 1
        end_time : datetime or str, optional
            end_time, by default 3600s or deduced from other parameters

        Returns
        -------
        pandas.DatetimeIndex
            time axis which can be used to create new Dataset

        Examples
        ------
        >>> t = Dateset.create_time('2018-1-1,2018-2-1', dt=1800)
        >>> t = Dateset.create_time('2018-1-1', dt=1800, n_timesteps=48)
        >>> t = Dateset.create_time('2018', dt=7200, end_time='2019')
        """
        if isinstance(start_time, str):
            parts = start_time.split(",")
            if len(parts) == 2:
                end_time = parts[1]
            start_time = pd.to_datetime(parts[0])
        if isinstance(end_time, str):
            end_time = pd.to_datetime(end_time)

        if start_time is None:
            # start_time = datetime.now()
            start_time = datetime(1970, 1, 1, 0, 0, 0)

        if dt is None:
            if (end_time is not None) and (n_timesteps is not None):
                dur = (end_time - start_time).total_seconds()
                dt = 0.0
                if (dur > 0) and (n_timesteps > 1):
                    dt = dur / (n_timesteps - 1)
            else:
                warnings.warn("Too little information. Assuming dt=3600s.")
                dt = 3600

        if (end_time is None) and (n_timesteps is None):
            warnings.warn("Too little information. Assuming n_timesteps=1.")
            n_timesteps = 1

        # find end time
        if end_time is None:
            tot_seconds = (n_timesteps - 1) * dt
            end_time = start_time + timedelta(seconds=tot_seconds)
        elif (end_time is not None) and (n_timesteps is not None):
            # both are given, do they match?
            tot_seconds = (n_timesteps - 1) * dt
            end_time_2 = start_time + timedelta(seconds=tot_seconds)
            if end_time != end_time_2:
                raise ValueError("All parameters where given, but they do not match")

        if dt < 0:
            raise ValueError("dt cannot be negative")

        if (end_time - start_time).total_seconds() < 0:
            raise ValueError("end_time must be greater than start_time")

        offset = pd.tseries.offsets.DateOffset(seconds=dt)
        return pd.date_range(start=start_time, end=end_time, freq=offset)

    @property
    def is_equidistant(self):
        """Is Dataset equidistant in time?"""
        if len(self.time) < 3:
            return True
        return len(self.time.to_series().diff().dropna().unique()) == 1
        # return self.time.freq is not None

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
    def n_timesteps(self):
        """Number of time steps"""
        return len(self.time)

    @property
    def n_items(self):
        """Number of items"""
        return len(self.items)

    @property
    def items(self):
        return self._items

    @property
    def shape(self):
        """Shape of each item"""
        return self.data[self._first_non_z_item].shape

    @property
    def _first_non_z_item(self):
        if len(self) > 1 and self.items[0].name == "Z coordinate":
            return 1
        return 0

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
