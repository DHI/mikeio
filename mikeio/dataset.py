import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from copy import deepcopy
from mikeio.eum import ItemInfo


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

        out = ["<mikeio.DataSet>"]
        out.append(f"Dimensions: {self.shape}")
        out.append(f"Time: {self.time[0]} - {self.time[-1]}")
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
        self, dt, method="linear", extrapolate=True, fill_value=np.nan,
    ):
        """Temporal interpolation

        Wrapper of `scipy.interpolate.interp`

        Parameters
        ----------
        dt: float or pd.DatetimeIndex
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

    @property
    def is_equidistant(self):
        """Is Dataset equidistant in time?
        """
        if len(self.time) < 3:
            return True

        return self.time.freq is not None

    @property
    def n_timesteps(self):
        """Number of time steps
        """
        return len(self.time)

    @property
    def n_items(self):
        """Number of items
        """
        return len(self.items)

    @property
    def shape(self):
        """Shape of each item 
        """
        return self.data[self._first_non_z_item].shape

    @property
    def _first_non_z_item(self):
        if len(self) > 1 and self.items[0].name == "Z coordinate":
            return 1
        return 0

    @property
    def n_elements(self):
        """Number of spatial elements/points
        """
        n_elem = np.prod(self.shape)
        if self.n_timesteps > 1:
            n_elem = int(n_elem / self.n_timesteps)
        return n_elem

