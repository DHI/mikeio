import warnings
from typing import Iterable, Optional, Sequence, Tuple, Union, Mapping
import numpy as np
import pandas as pd
from copy import deepcopy
from mikeio.eum import EUMType, EUMUnit, ItemInfo

from .spatial.grid_geometry import Grid1D, Grid2D
from .spatial.FM_geometry import GeometryFM, GeometryFMLayered
from .spatial.FM_utils import _plot_map

from .base import TimeSeries
from .spatial.geometry import _Geometry

# TODO use for dataset as well
def _parse_axis(data_shape, axis):
    axis = 0 if axis == "time" else axis
    if (axis == "spatial") or (axis == "space"):
        if len(data_shape) == 1:
            raise ValueError(
                f"axis '{axis}' not allowed for Dataset with shape {data_shape}"
            )
        axis = 1 if (len(data_shape) == 2) else tuple(range(1, len(data_shape)))
    if axis is None:
        axis = 0 if (len(data_shape) == 1) else tuple(range(0, len(data_shape)))
    if isinstance(axis, str):
        raise ValueError(
            f"axis argument '{axis}' not supported! Must be None, int, list of int or 'time' or 'space'"
        )
    return axis


def _time_by_axis(
    time: pd.DatetimeIndex, axis: Union[int, Sequence[int]]
) -> pd.DatetimeIndex:
    if axis == 0:
        time = pd.DatetimeIndex([time[0]])
    elif isinstance(axis, Sequence) and 0 in axis:
        time = pd.DatetimeIndex([time[0]])
    else:
        time = time

    return time


def _is_boolean_mask(x):
    if isinstance(x, np.ndarray):
        return x.dtype == np.dtype("bool")
    return False


def _get_by_boolean_mask(data: np.ndarray, mask: np.ndarray):
    if data.shape != mask.shape:
        return data[np.broadcast_to(mask, data.shape)]
    return data[mask]


def _set_by_boolean_mask(data: np.ndarray, mask: np.ndarray, value):
    if data.shape != mask.shape:
        data[np.broadcast_to(mask, data.shape)] = value
    else:
        data[mask] = value
    return


class _DataArrayPlotter:
    def __init__(self, da: "DataArray") -> None:
        self.da = da

    def __call__(self, ax=None, figsize=None, **kwargs):
        fig, ax = self._get_fig_ax(ax, figsize)

        if self.da.ndim == 1:
            return self._timeseries(self.da.values, fig, ax, **kwargs)

        if self.da.ndim == 2:
            return ax.imshow(self.da.values, **kwargs)

        # if everything else fails, plot histogram
        return self._hist(ax, **kwargs)

    @staticmethod
    def _get_ax(ax=None, figsize=None):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        return ax

    @staticmethod
    def _get_fig_ax(ax=None, figsize=None):
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = plt.gcf()
        return fig, ax

    def hist(self, ax=None, figsize=None, **kwargs):
        ax = self._get_ax(ax, figsize)
        return self._hist(ax, **kwargs)

    def _hist(self, ax, **kwargs):
        result = ax.hist(self.da.values.ravel(), **kwargs)
        ax.set_xlabel(self._label_txt())
        return result

    def line(self, ax=None, figsize=None, **kwargs):
        fig, ax = self._get_fig_ax(ax, figsize)
        return self._timeseries(self.da.values, fig, ax, **kwargs)

    def _timeseries(self, values, fig, ax, **kwargs):
        if "title" in kwargs:
            title = kwargs.pop("title")
            ax.set_title(title)
        ax.plot(self.da.time, values, **kwargs)
        ax.set_xlabel("time")
        fig.autofmt_xdate()
        ax.set_ylabel(self._label_txt())
        return ax

    def _label_txt(self):
        return f"{self.da.name} [{self.da.unit.name}]"


class _DataArrayPlotterGrid1D(_DataArrayPlotter):
    def __init__(self, da: "DataArray") -> None:
        super().__init__(da)

    def __call__(self, ax=None, figsize=None, **kwargs):
        ax = self._get_ax(ax, figsize)
        return self._lines(ax, **kwargs)

    def timeseries(self, ax=None, figsize=None, **kwargs):
        fig, ax = self._get_fig_ax(ax, figsize)
        return super()._timeseries(self.da.values, fig, ax, **kwargs)

    def imshow(self, ax=None, figsize=None, **kwargs):
        fig, ax = self._get_fig_ax(ax, figsize)
        pos = ax.imshow(self.da.values, **kwargs)
        fig.colorbar(pos, ax=ax, label=self._label_txt())
        return ax

    def pcolormesh(self, ax=None, figsize=None, **kwargs):
        fig, ax = self._get_fig_ax(ax, figsize)
        pos = ax.pcolormesh(
            self.da.geometry.x,
            self.da.time,
            self.da.values,
            shading="nearest",
            **kwargs,
        )
        cbar = fig.colorbar(pos, label=self._label_txt())
        ax.set_xlabel("x")
        ax.set_ylabel("time")
        return ax

    def _lines(self, ax=None, **kwargs):
        if "title" in kwargs:
            title = kwargs.pop("title")
            ax.set_title(title)
        ax.plot(self.da.geometry.x, self.da.values.T, **kwargs)
        ax.set_xlabel("x")
        ax.set_ylabel(self._label_txt())
        return ax


class _DataArrayPlotterGrid2D(_DataArrayPlotter):
    def __init__(self, da: "DataArray") -> None:
        super().__init__(da)

    def __call__(self, ax=None, figsize=None, **kwargs):
        return self.pcolormesh(ax, figsize, **kwargs)

    def contour(self, ax=None, figsize=None, **kwargs):
        fig, ax = self._get_fig_ax(ax, figsize)

        x, y = self._get_x_y()
        values = self._get_first_step_values()

        pos = ax.contour(x, y, np.flipud(values), **kwargs)
        # fig.colorbar(pos, label=self._label_txt())
        ax.clabel(pos, fmt="%1.2f", inline=1, fontsize=9)
        self._set_aspect_and_labels(ax, self.da.geometry.is_geo, y)
        return ax

    def contourf(self, ax=None, figsize=None, **kwargs):
        fig, ax = self._get_fig_ax(ax, figsize)

        x, y = self._get_x_y()
        values = self._get_first_step_values()

        pos = ax.contourf(x, y, np.flipud(values), **kwargs)
        fig.colorbar(pos, label=self._label_txt())
        self._set_aspect_and_labels(ax, self.da.geometry.is_geo, y)
        return ax

    def pcolormesh(self, ax=None, figsize=None, **kwargs):
        fig, ax = self._get_fig_ax(ax, figsize)

        xn, yn = self._get_xn_yn()
        values = self._get_first_step_values()

        pos = ax.pcolormesh(xn, yn, np.flipud(values), **kwargs)
        fig.colorbar(pos, label=self._label_txt())
        self._set_aspect_and_labels(ax, self.da.geometry.is_geo, yn)
        return ax

    def _get_first_step_values(self):
        if self.da.n_timesteps > 1:
            # select first step as default plotting behaviour
            return self.da.values[0]
        else:
            return np.squeeze(self.da.values)

    def _get_x_y(self):
        x = self.da.geometry.x
        y = self.da.geometry.y
        x = x + self.da.geometry._origin[0]
        y = y + self.da.geometry._origin[1]
        return x, y

    def _get_xn_yn(self):
        xn = self.da.geometry._centers_to_nodes(self.da.geometry.x)
        yn = self.da.geometry._centers_to_nodes(self.da.geometry.y)
        xn = xn + self.da.geometry._origin[0]
        yn = yn + self.da.geometry._origin[1]
        return xn, yn

    @staticmethod
    def _set_aspect_and_labels(ax, is_geo, y):
        if is_geo:
            ax.set_xlabel("Longitude [degrees]")
            ax.set_ylabel("Latitude [degrees]")
            mean_lat = np.mean(y)
            ax.set_aspect(1.0 / np.cos(np.pi * mean_lat / 180))
        else:
            ax.set_xlabel("Easting [m]")
            ax.set_ylabel("Northing [m]")
            ax.set_aspect("equal")


class _DataArrayPlotterFM(_DataArrayPlotter):
    def __init__(self, da: "DataArray") -> None:
        super().__init__(da)

    def __call__(self, ax=None, figsize=None, **kwargs):
        ax = self._get_ax(ax, figsize)
        return self._plot_FM_map(ax, **kwargs)

    def contour(self, ax=None, figsize=None, **kwargs):
        ax = self._get_ax(ax, figsize)
        kwargs["plot_type"] = "contour"
        return self._plot_FM_map(ax, **kwargs)

    def contourf(self, ax=None, figsize=None, **kwargs):
        ax = self._get_ax(ax, figsize)
        kwargs["plot_type"] = "contourf"
        return self._plot_FM_map(ax, **kwargs)

    def mesh(self, ax=None, figsize=None, **kwargs):
        return self.da.geometry.plot_mesh(figsize=figsize, ax=ax, **kwargs)

    def outline(self, ax=None, figsize=None, **kwargs):
        return self.da.geometry.plot_outline(figsize=figsize, ax=ax, **kwargs)

    def _plot_FM_map(self, ax, **kwargs):
        if self.da.n_timesteps > 1:
            # select first step as default plotting behaviour
            values = self.da.values[0]
        else:
            values = np.squeeze(self.da.values)

        title = f"{self.da.time[0]}"
        if self.da.geometry.is_2d:
            geometry = self.da.geometry
        else:
            # select surface as default plotting for 3d files
            values = values[self.da.geometry.top_elements]
            geometry = self.da.geometry.geometry2d
            title = "Surface, " + title

        if "label" not in kwargs:
            kwargs["label"] = self._label_txt()
        if "title" not in kwargs:
            kwargs["title"] = title

        return _plot_map(
            node_coordinates=geometry.node_coordinates,
            element_table=geometry.element_table,
            element_coordinates=geometry.element_coordinates,
            boundary_polylines=geometry.boundary_polylines,
            is_geo=geometry.is_geo,
            z=values,
            ax=ax,
            **kwargs,
        )


class DataArray(TimeSeries):

    deletevalue = 1.0e-35

    def __init__(
        self,
        data,
        # *,
        time: Union[pd.DatetimeIndex, str],
        item: ItemInfo = None,
        geometry: _Geometry = None,
        zn=None,
        dims: Optional[Sequence[str]] = None,
    ):

        data_lacks = []
        for p in ("shape", "ndim", "dtype"):
            if not hasattr(data, p):
                data_lacks.append(p)
        if len(data_lacks) > 0:
            raise TypeError(
                "Data must be ArrayLike, e.g. numpy array, but it lacks properties: "
                + ", ".join(data_lacks)
            )

        if dims is None:  # This is not very robust, but is probably a reasonable guess
            if data.ndim == 1:
                self.dims = ("t",)
            elif data.ndim == 2:
                self.dims = ("t", "x")
                # TODO FM
                # self.dims = ("t","e")
            elif data.ndim == 3:
                self.dims = ("t", "y", "x")
            elif data.ndim == 4:
                self.dims = ("t", "z", "y", "x")
        else:
            if data.ndim != len(dims):
                raise ValueError("Number of named dimensions does not equal data ndim")
            self.dims = dims

        self._values = data
        self.time = time
        if (item is not None) and (not isinstance(item, ItemInfo)):
            raise ValueError("Item must be an ItemInfo")
        self.item = item

        self.geometry = geometry

        if zn is not None:
            self._zn = zn

        if isinstance(geometry, GeometryFM):
            self.plot = _DataArrayPlotterFM(self)
        elif isinstance(geometry, Grid1D):
            self.plot = _DataArrayPlotterGrid1D(self)
        elif isinstance(geometry, Grid2D):
            self.plot = _DataArrayPlotterGrid2D(self)
        else:
            self.plot = _DataArrayPlotter(self)

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, value):
        if value.shape != self._values.shape:
            raise ValueError("Shape of new data is wrong")

        self._values = value

    def __setitem__(self, key, value):
        if _is_boolean_mask(key):
            return _set_by_boolean_mask(self._values, key, value)
        self._values[key] = value

    def __getitem__(self, key) -> "DataArray":
        if _is_boolean_mask(key):
            return _get_by_boolean_mask(self._values, key)

        if isinstance(key, tuple):
            steps = key[0]
        else:
            steps = key

        # select in time
        if isinstance(steps, str):
            steps = slice(steps, steps)
        if isinstance(steps, slice):
            try:
                s = self.time.slice_indexer(steps.start, steps.stop)
                steps = list(range(s.start, s.stop))
            except:
                steps = list(range(*steps.indices(self.n_timesteps)))
            time = self.time[steps]
        elif isinstance(steps, int):
            time = self.time[[steps]]
        else:
            time = self.time[steps]

        # select in space
        geometry = self.geometry
        zn = None
        if isinstance(key, tuple):
            if isinstance(self.geometry, GeometryFM):
                # TODO: allow for selection of layers
                elements = key[1]
                if isinstance(elements, slice):
                    elements = list(range(*elements.indices(self.geometry.n_elements)))
                else:
                    elements = np.atleast_1d(elements)
                if len(elements) == 1:
                    geometry = None
                else:
                    geometry = self.geometry.elements_to_geometry(elements)

                if isinstance(self.geometry, GeometryFMLayered):
                    nodes = self.geometry.element_table[elements]
                    unodes = np.unique(np.hstack(nodes))
                    zn = self._zn[:, unodes]

                key = (steps, elements)
            else:
                # TODO: better handling of dfs1,2,3
                key = (steps, *key[1:])
        else:
            key = steps

        data = self._values[key].copy()
        return DataArray(data=data, time=time, item=self.item, geometry=geometry, zn=zn)

    def to_numpy(self) -> np.ndarray:
        return self._values

    def flipud(self) -> "DataArray":
        """Flip upside down"""

        self.values = np.flip(self.values, axis=1)
        return self

    def max(self, axis="time") -> "DataArray":
        """Max value along an axis

        Parameters
        ----------
        axis: (int, str, None), optional
            axis number or "time" or "space", by default "time"=0

        Returns
        -------
        DataArray
            array with max values

        See Also
        --------
            nanmax : Max values with NaN values removed
        """
        return self.aggregate(axis=axis, func=np.max)

    def min(self, axis="time") -> "DataArray":
        """Min value along an axis

        Parameters
        ----------
        axis: (int, str, None), optional
            axis number or "time" or "space", by default "time"=0

        Returns
        -------
        DataArray
            array with min values

        See Also
        --------
            nanmin : Min values with NaN values removed
        """
        return self.aggregate(axis=axis, func=np.min)

    def mean(self, axis="time") -> "DataArray":
        """Mean value along an axis

        Parameters
        ----------
        axis: (int, str, None), optional
            axis number or "time" or "space", by default "time"=0

        Returns
        -------
        DataArray
            array with mean values

        See Also
        --------
            nanmean : Mean values with NaN values removed
        """
        return self.aggregate(axis=axis, func=np.mean)

    def nanmax(self, axis="time") -> "DataArray":
        """Max value along an axis (NaN removed)

        Parameters
        ----------
        axis: (int, str, None), optional
            axis number or "time" or "space", by default "time"=0

        Returns
        -------
        DataArray
            array with max values

        See Also
        --------
            nanmax : Max values with NaN values removed
        """
        return self.aggregate(axis=axis, func=np.nanmax)

    def nanmin(self, axis="time") -> "DataArray":
        """Min value along an axis (NaN removed)

        Parameters
        ----------
        axis: (int, str, None), optional
            axis number or "time" or "space", by default "time"=0

        Returns
        -------
        DataArray
            array with min values

        See Also
        --------
            nanmin : Min values with NaN values removed
        """
        return self.aggregate(axis=axis, func=np.nanmin)

    def nanmean(self, axis="time") -> "DataArray":
        """Mean value along an axis (NaN removed)

        Parameters
        ----------
        axis: (int, str, None), optional
            axis number or "time" or "space", by default "time"=0

        Returns
        -------
        DataArray
            array with mean values

        See Also
        --------
            mean : Mean values
        """
        return self.aggregate(axis=axis, func=np.nanmean)

    def aggregate(self, axis="time", func=np.nanmean, **kwargs) -> "DataArray":
        """Aggregate along an axis

        Parameters
        ----------
        axis: (int, str, None), optional
            axis number or "time" or "space", by default "time"=0
        func: function, optional
            default np.nanmean

        Returns
        -------
        DataArray
            dataarray with aggregated values

        See Also
        --------
            max : Max values
            nanmax : Max values with NaN values removed
        """

        axis = _parse_axis(self.shape, axis)
        time = _time_by_axis(self.time, axis)
        keepdims = axis == 0

        data = func(self.to_numpy(), axis=axis, keepdims=keepdims, **kwargs)

        if keepdims:
            geometry = self.geometry
        else:
            geometry = None
        return DataArray(data=data, time=time, item=self.item, geometry=geometry)

    def __radd__(self, other):
        return self.__add__(other)

    def __add__(self, other):
        if isinstance(other, self.__class__):
            return self._add_dataarray(other)
        else:
            return self._add_value(other)

    def __rsub__(self, other):
        ds = self.__mul__(-1.0)
        return other + ds

    def __sub__(self, other):
        if isinstance(other, self.__class__):
            return self._add_dataarray(other, sign=-1.0)
        else:
            return self._add_value(-other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            raise ValueError("Multiplication is not possible for two DataArrays")
        else:
            return self._multiply_value(other)

    def _add_dataarray(self, other, sign=1.0):
        # self._check_datasets_match(other) # TODO
        try:
            data = self.values + sign * other.values
        except:
            raise ValueError("Could not add data")

        new_da = self.copy()

        new_da.values = data

        return new_da

    def _add_value(self, value):
        try:
            data = value + self.values
        except:
            raise ValueError(f"{value} could not be added to DataArray")

        new_da = self.copy()

        new_da.values = data

        return new_da

    def _multiply_value(self, value):
        try:
            data = value * self.values
        except:
            raise ValueError(f"{value} could not be multiplied to DataArray")
        new_da = self.copy()

        new_da.values = data

        return new_da

    def copy(self):

        return deepcopy(self)

    @property
    def name(self) -> Optional[str]:
        if self.item.name:
            return self.item.name
        else:
            return None

    @name.setter
    def name(self, value):
        self.item.name = value

    @property
    def type(self) -> EUMType:
        return self.item.type

    @property
    def unit(self) -> EUMUnit:
        return self.item.unit

    @property
    def start_time(self):
        """First time instance (as datetime)"""
        return self.time[0].to_pydatetime()

    @property
    def end_time(self):
        """Last time instance (as datetime)"""
        return self.time[-1].to_pydatetime()

    @property
    def is_equidistant(self):
        """Is Dataset equidistant in time?"""
        if len(self.time) < 3:
            return True
        return len(self.time.to_series().diff().dropna().unique()) == 1

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
    def n_timesteps(self) -> int:
        """Number of time steps"""
        return len(self.time)

    @property
    def n_items(self) -> int:
        """Number of items"""
        return 1

    @property
    def items(self) -> Sequence[ItemInfo]:  # Sequence with a single element!
        return [self.item]

    @property
    def shape(self):
        return self.values.shape

    @property
    def ndim(self):
        return self.values.ndim

    @property
    def dtype(self):
        return self.values.dtype

    def __repr__(self):

        out = ["<mikeio.DataArray>"]
        if self.name is not None:
            out.append(f"Name: {self.name}")
        if isinstance(self.geometry, GeometryFM):
            gtxt = f"Geometry: {self.geometry.type_name}"
            if self.geometry.is_layered:
                n_z_layers = (
                    "no"
                    if self.geometry.n_z_layers is None
                    else self.geometry.n_z_layers
                )
                gtxt += f" ({self.geometry.n_sigma_layers} sigma-layers, {n_z_layers} z-layers)"

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

        return str.join("\n", out)
