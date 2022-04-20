from datetime import datetime
import warnings
from typing import Optional, Sequence, Tuple, Union, Iterable
import numpy as np
import pandas as pd
from copy import deepcopy

from .base import TimeSeries
from .eum import EUMType, EUMUnit, ItemInfo
from .spatial.geometry import (
    _Geometry,
    GeometryPoint2D,
    GeometryPoint3D,
    GeometryUndefined,
)
from .spatial.grid_geometry import Grid1D, Grid2D
from .spatial.FM_geometry import (
    GeometryFM,
    GeometryFMLayered,
    GeometryFMPointSpectrum,
    GeometryFMVerticalColumn,
    GeometryFMVerticalProfile,
)
from mikecore.DfsuFile import DfsuFileType
from .spatial.FM_utils import _plot_map, _plot_vertical_profile
from .data_utils import DataUtilsMixin


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

        pos = ax.contour(x, y, values, **kwargs)
        # fig.colorbar(pos, label=self._label_txt())
        ax.clabel(pos, fmt="%1.2f", inline=1, fontsize=9)
        self._set_aspect_and_labels(ax, self.da.geometry.is_geo, y)
        return ax

    def contourf(self, ax=None, figsize=None, **kwargs):
        fig, ax = self._get_fig_ax(ax, figsize)

        x, y = self._get_x_y()
        values = self._get_first_step_values()

        pos = ax.contourf(x, y, values, **kwargs)
        fig.colorbar(pos, label=self._label_txt())
        self._set_aspect_and_labels(ax, self.da.geometry.is_geo, y)
        return ax

    def pcolormesh(self, ax=None, figsize=None, **kwargs):
        fig, ax = self._get_fig_ax(ax, figsize)

        xn, yn = self._get_xn_yn()
        values = self._get_first_step_values()

        pos = ax.pcolormesh(xn, yn, values, **kwargs)
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
        # x = x + self.da.geometry._origin[0] # TODO WTF?
        # y = y + self.da.geometry._origin[1] # TODO
        return x, y

    def _get_xn_yn(self):
        xn = self.da.geometry._centers_to_nodes(self.da.geometry.x)
        yn = self.da.geometry._centers_to_nodes(self.da.geometry.y)
        # xn = xn + self.da.geometry._origin[0] # TODO
        # yn = yn + self.da.geometry._origin[1] # TODO
        return xn, yn

    @staticmethod
    def _set_aspect_and_labels(ax, is_geo, y):
        if is_geo:
            ax.set_xlabel("Longitude [degrees]")
            ax.set_ylabel("Latitude [degrees]")
            mean_lat = np.mean(y)
            aspect_ratio = 1.0 / np.cos(np.pi * mean_lat / 180)
            ax.set_aspect(aspect_ratio)
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
        return self.da.geometry.plot.mesh(figsize=figsize, ax=ax, **kwargs)

    def outline(self, ax=None, figsize=None, **kwargs):
        return self.da.geometry.plot.outline(figsize=figsize, ax=ax, **kwargs)

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


class _DataArrayPlotterFMVerticalColumn(_DataArrayPlotter):
    def __init__(self, da: "DataArray") -> None:
        super().__init__(da)

    def __call__(self, ax=None, figsize=None, **kwargs):
        ax = self._get_ax(ax, figsize)
        return self.lines(ax, **kwargs)

    def lines(self, ax=None, figsize=None, extrapolate=True, **kwargs):
        ax = self._get_ax(ax, figsize)
        return self._lines(ax, extrapolate=extrapolate, **kwargs)

    def _lines(self, ax=None, show_legend=None, extrapolate=True, **kwargs):
        import matplotlib.pyplot as plt

        if "title" in kwargs:
            title = kwargs.pop("title")
            ax.set_title(title)

        if show_legend is None:
            show_legend = len(self.da.time) < 10

        values = self.da.to_numpy()
        zn = self.da._zn
        if extrapolate:
            ze = self.da.geometry._calc_zee(zn)
            values = self.da.geometry._interp_values(zn, values, ze)
        else:
            ze = self.da.geometry.calc_ze(zn)

        ax.plot(values.T, ze.T, label=self.da.time, **kwargs)

        ax.set_xlabel(self._label_txt())
        ax.set_ylabel("z")

        if show_legend:
            plt.legend()

        return ax

    def pcolormesh(self, ax=None, figsize=None, **kwargs):
        fig, ax = self._get_fig_ax(ax, figsize)
        ze = self.da.geometry.calc_ze()
        pos = ax.pcolormesh(
            self.da.time,
            ze,
            self.da.values.T,
            shading="nearest",
            **kwargs,
        )
        cbar = fig.colorbar(pos, label=self._label_txt())
        ax.set_xlabel("time")
        fig.autofmt_xdate()
        ax.set_ylabel("z (static)")
        return ax


class _DataArrayPlotterFMVerticalProfile(_DataArrayPlotter):
    def __init__(self, da: "DataArray") -> None:
        super().__init__(da)

    def __call__(self, ax=None, figsize=None, **kwargs):
        ax = self._get_ax(ax, figsize)
        return self._plot_transect(ax=ax, **kwargs)

    def _plot_transect(self, **kwargs):
        if "label" not in kwargs:
            kwargs["label"] = self._label_txt()
        if "title" not in kwargs:
            kwargs["title"] = self.da.time[0]

        values, zn = self._get_first_step_values()
        g = self.da.geometry
        return _plot_vertical_profile(
            node_coordinates=g.node_coordinates,
            element_table=g.element_table,
            values=values,
            zn=zn,
            **kwargs,
        )

    def _get_first_step_values(self):
        if self.da.n_timesteps > 1:
            return self.da.values[0], self.da._zn[0]
        else:
            return np.squeeze(self.da.values), np.squeeze(self.da._zn)


class DataArray(DataUtilsMixin, TimeSeries):

    deletevalue = 1.0e-35

    def __init__(
        self,
        data,
        # *,
        time: Union[pd.DatetimeIndex, str] = None,
        item: ItemInfo = None,
        geometry: _Geometry = GeometryUndefined(),
        zn=None,
        dims: Optional[Sequence[str]] = None,
    ):
        # TODO: add optional validation validate=True
        self._values = self._parse_data(data)
        self.time = self._parse_time(time)
        self.dims = self._parse_dims(dims, geometry)

        self._check_time_data_length(self.time)

        self.item = self._parse_item(item)
        self.geometry = self._parse_geometry(geometry, self.dims, self.shape)
        self._zn = self._parse_zn(zn, self.geometry, self.n_timesteps)
        self.plot = self._get_plotter_by_geometry()

    @staticmethod
    def _parse_data(data):
        validation_errors = []
        for p in ("shape", "ndim", "dtype"):
            if not hasattr(data, p):
                validation_errors.append(p)
        if len(validation_errors) > 0:
            raise TypeError(
                "Data must be ArrayLike, e.g. numpy array, but it lacks properties: "
                + ", ".join(validation_errors)
            )
        return data

    def _parse_dims(self, dims, geometry):
        if dims is None:
            return self._guess_dims(self.ndim, self.shape, self.n_timesteps, geometry)
        else:
            if self.ndim != len(dims):
                raise ValueError("Number of named dimensions does not equal data ndim")
            if ("time" in dims) and dims[0] != "time":
                raise ValueError("time must be first dimension if present!")
            if (self.n_timesteps > 1) and ("time" not in dims):
                raise ValueError(
                    f"time missing from named dimensions {dims}! (number of timesteps: {self.n_timesteps})"
                )
            return tuple(dims)

    @staticmethod
    def _guess_dims(ndim, shape, n_timesteps, geometry):
        # This is not very robust, but is probably a reasonable guess
        time_is_first = (n_timesteps > 1) or (shape[0] == 1 and n_timesteps == 1)
        dims = ["time"] if time_is_first else []
        ndim_no_time = ndim if (len(dims) == 0) else ndim - 1

        if isinstance(geometry, GeometryFMPointSpectrum):
            if ndim_no_time > 0:
                dims.append("frequency")
            if ndim_no_time > 1:
                dims.append("direction")
        elif isinstance(geometry, GeometryFM):
            if geometry._type == DfsuFileType.DfsuSpectral1D:
                if ndim_no_time > 0:
                    dims.append("node")
            else:
                if ndim_no_time > 0:
                    dims.append("element")
            if geometry.is_spectral:
                if ndim_no_time > 1:
                    dims.append("frequency")
                if ndim_no_time > 2:
                    dims.append("direction")
        elif isinstance(geometry, Grid1D):
            dims.append("x")
        elif isinstance(geometry, Grid2D):
            dims.append("y")
            dims.append("x")
        else:
            # gridded
            if ndim_no_time > 2:
                dims.append("z")
            if ndim_no_time > 1:
                dims.append("y")
            if ndim_no_time > 0:
                dims.append("x")
        return tuple(dims)

    def _check_time_data_length(self, time):
        if "time" in self.dims and len(time) != self._values.shape[0]:
            raise ValueError(
                f"Number of timesteps ({len(time)}) does not fit with data shape {self.values.shape}"
            )

    @staticmethod
    def _parse_item(item):
        if item is None:
            return ItemInfo("NoName")

        if not isinstance(item, ItemInfo):
            try:
                item = ItemInfo(item)
            except:
                raise ValueError(
                    "Item must be None, ItemInfo or valid input to ItemInfo"
                )
        return item

    @staticmethod
    def _parse_geometry(geometry, dims, shape):
        if len(dims) > 1 and (
            geometry is None or isinstance(geometry, GeometryUndefined)
        ):
            # raise ValueError("Geometry is required for ndim >=1")
            warnings.warn("Geometry is required for ndim >=1")

        axis = 1 if "time" in dims else 0
        # dims_no_time = tuple([d for d in dims if d != "time"])
        # shape_no_time = shape[1:] if ("time" in dims) else shape

        if len(dims) == 1 and dims[0] == "time":
            if geometry is not None:
                # assert geometry.ndim == 0
                return geometry
            else:
                return GeometryUndefined()

        if isinstance(geometry, GeometryFMPointSpectrum):
            pass
        elif isinstance(geometry, GeometryFM):
            if geometry.is_spectral:
                if geometry._type == DfsuFileType.DfsuSpectral1D:
                    assert (
                        shape[axis] == geometry.n_nodes
                    ), "data shape does not match number of nodes"
                elif geometry._type == DfsuFileType.DfsuSpectral2D:
                    assert (
                        shape[axis] == geometry.n_elements
                    ), "data shape does not match number of elements"
            else:
                assert (
                    shape[axis] == geometry.n_elements
                ), "data shape does not match number of elements"
        elif isinstance(geometry, Grid1D):
            assert (
                shape[axis] == geometry.n
            ), "data shape does not match number of grid points"
        elif isinstance(geometry, Grid2D):
            assert shape[axis] == geometry.ny, "data shape does not match ny"
            assert shape[axis + 1] == geometry.nx, "data shape does not match nx"
        # elif isinstance(geometry, Grid3D): # TODO

        return geometry

    @staticmethod
    def _parse_zn(zn, geometry, n_timesteps):
        if zn is not None:
            if isinstance(geometry, GeometryFMLayered):
                # TODO: np.squeeze(zn) if n_timesteps=1 ?
                if (n_timesteps > 1) and (zn.shape[0] != n_timesteps):
                    raise ValueError(
                        f"zn has wrong shape ({zn.shape}). First dimension should be of size n_timesteps ({n_timesteps})"
                    )
                if zn.shape[-1] != geometry.n_nodes:
                    raise ValueError(
                        f"zn has wrong shape ({zn.shape}). Last dimension should be of size n_nodes ({geometry.n_nodes})"
                    )
            else:
                raise ValueError("zn can only be provided for layered dfsu data")
        return zn

    def _is_compatible(self, other, raise_error=False):
        """check if other DataArray has equivalent dimensions, time and geometry"""
        problems = []
        if not isinstance(other, DataArray):
            return False
        if self.shape != other.shape:
            problems.append("shape of data must be the same")
        if self.n_timesteps != other.n_timesteps:
            problems.append("Number of timesteps must be the same")
        if self.start_time != other.start_time:
            problems.append("start_time must be the same")
        if type(self.geometry) != type(other.geometry):
            problems.append("The type of geometry must be the same")
        if hasattr(self.geometry, "__eq__"):
            if not (self.geometry == self.geometry):
                problems.append("The geometries must be the same")
        if self._zn is not None:
            # it can be expensive to check equality of zn
            # so we test only size, first and last element
            if (
                other._zn is None
                or self._zn.shape != other._zn.shape
                or self._zn.ravel()[0] != other._zn.ravel()[0]
                or self._zn.ravel()[-1] != other._zn.ravel()[-1]
            ):
                problems.append("zn must be the same")

        if self.dims != other.dims:
            problems.append("Dimension names (dims) must be the same")

        if raise_error and len(problems) > 0:
            raise ValueError(", ".join(problems))

        return len(problems) == 0

    def _get_plotter_by_geometry(self):
        if isinstance(self.geometry, GeometryFMVerticalProfile):
            return _DataArrayPlotterFMVerticalProfile(self)
        elif isinstance(self.geometry, GeometryFMVerticalColumn):
            return _DataArrayPlotterFMVerticalColumn(self)
        elif isinstance(self.geometry, GeometryFM):
            return _DataArrayPlotterFM(self)
        elif isinstance(self.geometry, Grid1D):
            return _DataArrayPlotterGrid1D(self)
        elif isinstance(self.geometry, Grid2D):
            return _DataArrayPlotterGrid2D(self)
        else:
            return _DataArrayPlotter(self)

    # ============= Basic properties/methods ===========

    @property
    def name(self) -> Optional[str]:
        return self.item.name

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
        # TODO: use pd.Timestamp instead
        return self.time[0].to_pydatetime()

    @property
    def end_time(self):
        """Last time instance (as datetime)"""
        # TODO: use pd.Timestamp instead
        return self.time[-1].to_pydatetime()

    @property
    def is_equidistant(self) -> bool:
        """Is Dataset equidistant in time?"""
        if len(self.time) < 3:
            return True
        return len(self.time.to_series().diff().dropna().unique()) == 1

    @property
    def timestep(self) -> Optional[float]:
        """Time step in seconds if equidistant (and at
        least two time instances); otherwise None
        """
        dt = None
        if len(self.time) > 1 and self.is_equidistant:
            dt = (self.time[1] - self.time[0]).total_seconds()
        return dt

    @property
    def n_timesteps(self) -> int:
        """Number of time steps"""
        return len(self.time)

    @property
    def shape(self):
        return self.values.shape

    @property
    def ndim(self) -> int:
        return self.values.ndim

    @property
    def dtype(self):
        return self.values.dtype

    @property
    def values(self) -> np.ndarray:
        return self._values

    @values.setter
    def values(self, value):
        if value.shape != self._values.shape:
            raise ValueError("Shape of new data is wrong")

        self._values = value

    def to_numpy(self) -> np.ndarray:
        return self._values

    def flipud(self) -> "DataArray":
        """Flip upside down"""

        # TODO: axis=self._first_non_time_axis
        self.values = np.flip(self.values, axis=1)
        return self

    def describe(self, **kwargs) -> pd.DataFrame:
        """Generate descriptive statistics by wrapping pandas describe()"""
        data = {}
        data[self.name] = self.to_numpy().ravel()
        df = pd.DataFrame(data).describe(**kwargs)

        return df

    def copy(self) -> "DataArray":
        return deepcopy(self)

    def squeeze(self) -> "DataArray":
        """
        Remove axes of length 1

        Returns
        -------
        DataArray
        """

        data = np.squeeze(self.values)

        dims = [d for s, d in zip(self.shape, self.dims) if s != 1]

        # TODO: should geometry stay the same?
        return DataArray(
            data=data,
            time=self.time,
            item=self.item,
            geometry=self.geometry,
            zn=self._zn,
            dims=tuple(dims),
        )

    # ============= Select/interp ===========

    def __getitem__(self, key) -> "DataArray":
        if self._is_boolean_mask(key):
            mask = key if isinstance(key, np.ndarray) else key.values
            return self._get_by_boolean_mask(self.values, mask)

        da = self
        dims = self.dims
        key = self._getitem_parse_key(key)
        for j, k in enumerate(key):
            if isinstance(k, Iterable) or k != slice(None):
                if dims[j] == "time":
                    # getitem accepts fancy indexing only for time
                    k = self._get_time_idx_list(self.time, k)
                    if len(k) == 0:
                        raise IndexError("No timesteps found!")
                da = da.isel(k, axis=dims[j])
        return da

    def _getitem_parse_key(self, key):
        if isinstance(key, tuple):
            # is it multiindex or just a tuple of indexes for first axis?
            # da[2,3,4] and da[(2,3,4)] both have the key=(2,3,4)
            # how do we know if user wants step 2,3,4 or t=2,y=3,x=4 ?
            all_idx_int = True
            any_idx_after_0_time = False
            for j, k in enumerate(key):
                if not isinstance(k, int):
                    all_idx_int = False
                if j >= 1 and isinstance(k, (str, pd.Timestamp, datetime)):
                    any_idx_after_0_time = True
            if all_idx_int and (len(key) > self.ndim):
                if np.all(np.diff(key) >= 1):
                    # tuple with increasing list of indexes larger than the number of dims
                    key = (list(key),)
            if any_idx_after_0_time and self.dims[0][0] == "t":
                # tuple of times, must refer to time axis
                key = (list(key),)

        key = key if isinstance(key, tuple) else (key,)
        if len(key) > len(self.dims):
            raise IndexError(
                f"Key has more dimensions ({len(key)}) than DataArray ({len(self.dims)})!"
            )
        return key

    def __setitem__(self, key, value):
        if self._is_boolean_mask(key):
            mask = key if isinstance(key, np.ndarray) else key.values
            return self._set_by_boolean_mask(self._values, mask, value)
        self._values[key] = value

    def isel(self, idx=None, axis=0, **kwargs) -> "DataArray":
        """
        Select subset along an axis.

        Parameters
        ----------
        idx: int, scalar or array_like
        axis: (int, str, None), optional
            axis number or "time", by default 0

        Returns
        -------
        DataArray
            data with subset

        """
        for dim in kwargs:
            if dim in self.dims:
                axis = dim
                if idx is not None:
                    raise NotImplementedError(
                        "Selecting on multiple dimensions in the same call, not yet implemented"
                    )
                idx = kwargs[dim]
            else:
                raise ValueError(f"{dim} is not present in {self.dims}")

        axis = self._parse_axis(self.shape, self.dims, axis)

        if isinstance(idx, slice):
            idx = list(range(*idx.indices(self.shape[axis])))
        if idx is None or (not np.isscalar(idx) and len(idx) == 0):
            return None

        idx = np.atleast_1d(idx)
        single_index = len(idx) == 1
        idx = idx[0] if single_index else idx

        if axis == 0 and self.dims[0] == "time":
            time = self.time[idx]
            geometry = self.geometry
            zn = None if self._zn is None else self._zn[idx]
        else:
            time = self.time
            geometry = GeometryUndefined()
            zn = None
            if hasattr(self.geometry, "isel"):
                spatial_axis = self._axis_to_spatial_axis(self.dims, axis)
                geometry = self.geometry.isel(idx, axis=spatial_axis)
            if isinstance(geometry, GeometryFMLayered):
                node_ids, _ = self.geometry._get_nodes_and_table_for_elements(
                    idx, node_layers="all"
                )
                zn = self._zn[:, node_ids]

        if single_index:
            # reduce dims only if singleton idx
            dims = tuple([d for i, d in enumerate(self.dims) if i != axis])
            dat = np.take(self.values, int(idx), axis=axis)
        else:
            dims = self.dims
            dat = np.take(self.values, idx, axis=axis)

        return DataArray(
            data=dat,
            time=time,
            item=deepcopy(self.item),
            geometry=geometry,
            zn=zn,
            dims=dims,
        )

    def sel(
        self,
        *,
        time: Union[str, pd.DatetimeIndex, "DataArray"] = None,
        x: float = None,
        y: float = None,
        z: float = None,
        **kwargs,
    ) -> "DataArray":

        # TODO: delegate select in space to geometry
        t_ax = 1 if self.dims[0][0] == "t" else 0

        # select in space
        if (x is not None) or (y is not None) or (z is not None):
            if isinstance(self.geometry, Grid2D):  # TODO find out better way
                xy = np.column_stack((x, y))
                if len(xy) > 1:
                    raise NotImplementedError(
                        "Grid2D does not support multiple point sel()"
                    )
                i, j = self.geometry.find_index(xy=xy)
                if i == -1 or j == -1:
                    return None
                tmp = self.isel(idx=j[0], axis=(0 + t_ax))
                sp_axis = 0 if len(j) == 1 else 1
                da = tmp.isel(idx=i[0], axis=(sp_axis + t_ax))
            else:
                idx = self.geometry.find_index(x=x, y=y, z=z)
                da = self.isel(idx, axis="space")
        else:
            da = self

        if "layer" in kwargs:
            if isinstance(da.geometry, GeometryFMLayered):
                layer = kwargs.pop("layer")
                idx = da.geometry.get_layer_elements(layer)
                da = da.isel(idx, axis="space")
            else:
                raise ValueError("'layer' can only be selected from layered Dfsu data")

        if "area" in kwargs:
            area = kwargs.pop("area")
            if isinstance(da.geometry, GeometryFM):
                idx = da.geometry._elements_in_area(area)
                da = da.isel(idx, axis="space")
            elif isinstance(da.geometry, Grid2D):
                ii, jj = self.geometry.find_index(area=area)
                tmp = self.isel(idx=jj, axis=(0 + t_ax))
                sp_axis = 0 if len(jj) == 1 else 1
                da = tmp.isel(idx=ii, axis=(sp_axis + t_ax))
            else:
                raise ValueError(
                    "'area' can only be selected from Grid2D or flexible mesh data"
                )

        if len(kwargs) > 0:
            args = ",".join(kwargs)
            raise ValueError(f"Argument(s) '{args}' not recognized (layer, area).")

        # select in time
        if time is not None:
            time = time.time if isinstance(time, TimeSeries) else time
            if isinstance(time, int) or (
                isinstance(time, Sequence) and isinstance(time[0], int)
            ):
                da = da.isel(time, axis="time")
            else:
                da = da[time]

        return da

    def interp(
        # TODO find out optimal syntax to allow interpolation to single point, new time, grid, mesh...
        self,
        *,
        time: Union[pd.DatetimeIndex, "DataArray"] = None,
        x: float = None,
        y: float = None,
        z: float = None,
        n_nearest=3,
        interpolant=None,
        **kwargs,
    ) -> "DataArray":

        if z is not None:
            raise NotImplementedError()

        # interp in space
        if (x is not None) or (y is not None) or (z is not None):
            xy = [(x, y)]

            if isinstance(self.geometry, Grid2D):  # TODO DIY bilinear interpolation
                xr_da = self.to_xarray()
                dai = xr_da.interp(x=x, y=y).values
                geometry = GeometryPoint2D(x=x, y=y)
            elif isinstance(self.geometry, Grid1D):
                if interpolant is None:
                    interpolant = self.geometry.get_spatial_interpolant(xy)
                dai = self.geometry.interp(self.to_numpy(), *interpolant).flatten()
                geometry = GeometryUndefined()
            elif isinstance(self.geometry, GeometryFM):
                if interpolant is None:
                    interpolant = self.geometry.get_2d_interpolant(
                        xy, n_nearest=n_nearest, **kwargs
                    )
                dai = self.geometry.interp2d(self.to_numpy(), *interpolant).flatten()
                if z is None:
                    geometry = GeometryPoint2D(x=x, y=y)
                else:
                    geometry = GeometryPoint3D(x=x, y=y, z=z)

            da = DataArray(
                data=dai, time=self.time, geometry=geometry, item=deepcopy(self.item)
            )
        else:
            da = self.copy()

        # interp in time
        if time is not None:
            da = da.interp_time(time)

        return da

    def interp_time(
        self,
        dt: Union[float, pd.DatetimeIndex, "DataArray"],
        *,
        method="linear",
        extrapolate=True,
        fill_value=np.nan,
    ) -> "DataArray":
        """Temporal interpolation

        Wrapper of `scipy.interpolate.interp`

        Parameters
        ----------
        dt: float or pd.DatetimeIndex or Dataset/DataArray
            output timestep in seconds or new time axis
        method: str or int, optional
            Specifies the kind of interpolation as a string ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', 'next', where 'zero', 'slinear', 'quadratic' and 'cubic' refer to a spline interpolation of zeroth, first, second or third order; 'previous' and 'next' simply return the previous or next value of the point) or as an integer specifying the order of the spline interpolator to use. Default is 'linear'.
        extrapolate: bool, optional
            Default True. If False, a ValueError is raised any time interpolation is attempted on a value outside of the range of x (where extrapolation is necessary). If True, out of bounds values are assigned fill_value
        fill_value: float or array-like, optional
            Default NaN. this value will be used to fill in for points outside of the time range.

        Returns
        -------
        DataArray
        """
        t_out_index = self._parse_interp_time(self.time, dt)
        t_in = self.time.values.astype(float)
        t_out = t_out_index.values.astype(float)

        data = self._interpolate_time(
            t_in, t_out, self.to_numpy(), method, extrapolate, fill_value
        )

        zn = (
            None
            if self._zn is None
            else self._interpolate_time(
                t_in, t_out, self._zn, method, extrapolate, fill_value
            )
        )

        return DataArray(
            data,
            t_out_index,
            item=deepcopy(self.item),
            geometry=self.geometry,
            zn=zn,
        )

    def interp_like(
        self,
        other: Union["DataArray", Grid2D, GeometryFM, pd.DatetimeIndex],
        interpolant=None,
        **kwargs,
    ) -> "DataArray":
        """Interpolate in space (and in time) to other geometry (and time axis)

        Note: currently only supports interpolation from dfsu-2d to
              dfs2 or other dfsu-2d DataArrays

        Parameters
        ----------
        other: Dataset, DataArray, Grid2D, GeometryFM, pd.DatetimeIndex
        interpolant, optional
            Reuse pre-calculated index and weights
        kwargs: additional kwargs are passed to interpolation method

        Examples
        --------
        >>> dai = da.interp_like(da2)
        >>> dae = da.interp_like(da2, extrapolate=True)
        >>> dat = da.interp_like(da2.time)

        Returns
        -------
        DataArray
            Interpolated DataArray
        """
        if isinstance(other, pd.DatetimeIndex):
            return self.interp_time(other, **kwargs)

        if not (isinstance(self.geometry, GeometryFM) and self.geometry.is_2d):
            raise NotImplementedError("Currently only supports 2d flexible mesh data!")

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

        if interpolant is None:
            interpolant = self.geometry.get_2d_interpolant(xy, **kwargs)

        if isinstance(geom, Grid2D):
            dai = self.geometry.interp2d(
                self.to_numpy(), *interpolant, shape=(geom.ny, geom.nx)
            )
        else:
            dai = self.geometry.interp2d(self.to_numpy(), *interpolant)

        dai = DataArray(
            data=dai, time=self.time, geometry=geom, item=deepcopy(self.item)
        )

        if hasattr(other, "time"):
            dai = dai.interp_time(other.time)

        return dai

    # ============= Aggregation methods ===========

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

        axis = self._parse_axis(self.shape, self.dims, axis)
        time = self._time_by_agg_axis(self.time, axis)

        if isinstance(axis, Iterable):
            dims = tuple([d for i, d in enumerate(self.dims) if i not in axis])
        else:
            dims = tuple([d for i, d in enumerate(self.dims) if i != axis])

        data = func(self.to_numpy(), axis=axis, keepdims=False, **kwargs)

        if axis == 0:  # time
            geometry = self.geometry
            zn = None if self._zn is None else self._zn[0]

        else:
            geometry = GeometryUndefined()
            zn = None

        return DataArray(
            data=data,
            time=time,
            item=deepcopy(self.item),
            geometry=geometry,
            dims=dims,
            zn=zn,
        )

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
        DataArray
            data with quantile values

        Examples
        --------
        >>> da.quantile(q=[0.25,0.75])
        >>> da.quantile(q=0.5)
        >>> da.quantile(q=[0.01,0.5,0.99], axis="space")

        See Also
        --------
        nanquantile : quantile with NaN values ignored
        """
        return self._quantile(q, axis=axis, func=np.quantile, **kwargs)

    def _quantile(self, q, *, axis=0, func=np.quantile, **kwargs):

        from mikeio import Dataset

        axis = self._parse_axis(self.shape, self.dims, axis)
        time = self._time_by_agg_axis(self.time, axis)

        if np.isscalar(q):
            qdat = func(self.values, q=q, axis=axis, **kwargs)
            geometry = self.geometry if axis == 0 else GeometryUndefined()
            zn = self._zn if axis == 0 else None

            dims = tuple([d for i, d in enumerate(self.dims) if i != axis])
            item = deepcopy(self.item)
            return DataArray(qdat, time, item=item, geometry=geometry, dims=dims, zn=zn)
        else:
            res = []
            for quantile in q:
                qd = self._quantile(q=quantile, axis=axis, func=func)
                newname = f"Quantile {quantile}, {self.name}"
                qd.name = newname
                res.append(qd)

            return Dataset(res)

    # ============= MATH operations ===========

    def __radd__(self, other) -> "DataArray":
        return self.__add__(other)

    def __add__(self, other) -> "DataArray":
        return self._apply_math_operation(other, np.add, "+")

    def __rsub__(self, other) -> "DataArray":
        return other + self.__neg__()

    def __sub__(self, other) -> "DataArray":
        return self._apply_math_operation(other, np.subtract, "-")

    def __rmul__(self, other) -> "DataArray":
        return self.__mul__(other)

    def __mul__(self, other) -> "DataArray":
        return self._apply_math_operation(other, np.multiply, "*")

    def __pow__(self, other) -> "DataArray":
        return self._apply_math_operation(other, np.power, "**")

    def __truediv__(self, other) -> "DataArray":
        return self._apply_math_operation(other, np.divide, "/")

    def __floordiv__(self, other) -> "DataArray":
        return self._apply_math_operation(other, np.floor_divide, "//")

    def __mod__(self, other) -> "DataArray":
        return self._apply_math_operation(other, np.mod, "%")

    def __neg__(self) -> "DataArray":
        return self._apply_unary_math_operation(np.negative)

    def __pos__(self) -> "DataArray":
        return self._apply_unary_math_operation(np.positive)

    def __abs__(self) -> "DataArray":
        return self._apply_unary_math_operation(np.abs)

    def _apply_unary_math_operation(self, func) -> "DataArray":
        try:
            data = func(self.values)
        except:
            # TODO: better except... TypeError etc
            raise ValueError(f"Math operation could not be applied to DataArray")

        new_da = self.copy()
        new_da.values = data
        return new_da

    def _apply_math_operation(self, other, func, txt="with") -> "DataArray":
        """Apply a binary math operation with a scalar, an array or another DataArray"""
        try:
            other_values = other.values if hasattr(other, "values") else other
            data = func(self.values, other_values)
        except:
            # TODO: better except... TypeError etc
            raise ValueError(f"Math operation could not be applied to DataArray")

        # TODO: check if geometry etc match if other is DataArray?

        new_da = self.copy()  # TODO: alternatively: create new dataset (will validate)
        new_da.values = data

        if not self._keep_EUM_after_math_operation(other, func):
            other_name = other.name if hasattr(other, "name") else "array"
            new_da.item = ItemInfo(
                f"{self.name} {txt} {other_name}", itemtype=EUMType.Undefined
            )

        return new_da

    def _keep_EUM_after_math_operation(self, other, func) -> bool:
        """Does the math operation falsify the EUM?"""
        if hasattr(other, "shape") and hasattr(other, "ndim"):
            # other is array-like, so maybe we cannot keep EUM
            if func == np.subtract or func == np.sum:
                # +/-: we may want to keep EUM
                if isinstance(other, DataArray):
                    if self.type == other.type and self.unit == other.unit:
                        return True
                    else:
                        return False
                else:
                    return True  # assume okay, since no EUM
            return False

        # other is likely scalar, okay to keep EUM
        return True

    # ============= Logical indexing ===========

    def __lt__(self, other) -> "DataArray":
        bmask = self.values < self._other_to_values(other)
        return self._boolmask_to_new_DataArray(bmask)

    def __gt__(self, other) -> "DataArray":
        bmask = self.values > self._other_to_values(other)
        return self._boolmask_to_new_DataArray(bmask)

    def __le__(self, other) -> "DataArray":
        bmask = self.values <= self._other_to_values(other)
        return self._boolmask_to_new_DataArray(bmask)

    def __ge__(self, other) -> "DataArray":
        bmask = self.values >= self._other_to_values(other)
        return self._boolmask_to_new_DataArray(bmask)

    def __eq__(self, other) -> "DataArray":
        bmask = self.values == self._other_to_values(other)
        return self._boolmask_to_new_DataArray(bmask)

    def __ne__(self, other) -> "DataArray":
        bmask = self.values != self._other_to_values(other)
        return self._boolmask_to_new_DataArray(bmask)

    @staticmethod
    def _other_to_values(other):
        return other.values if isinstance(other, DataArray) else other

    def _boolmask_to_new_DataArray(self, bmask) -> "DataArray":
        return DataArray(
            data=bmask,
            time=self.time,
            item=ItemInfo("Boolean"),
            geometry=self.geometry,
            zn=self._zn,
        )

    # ============= output methods: to_xxx() ===========

    def _to_dataset(self):
        """Create a single-item dataset"""
        from mikeio import Dataset

        return Dataset(
            {self.name: self}
        )  # Single-item Dataset (All info is contained in the DataArray, no need for additional info)

    def to_dfs(self, filename) -> None:
        self._to_dataset().to_dfs(filename)

    def to_xarray(self):
        import xarray as xr

        coords = None

        if isinstance(self.geometry, Grid2D):
            coords = {}
            coords["time"] = xr.DataArray(self.time, dims="time")
            coords["x"] = xr.DataArray(data=self.geometry.x, dims="x")
            coords["y"] = xr.DataArray(data=self.geometry.y, dims="y")

        # TODO other geometries

        xr_da = xr.DataArray(
            data=self.values,
            name=self.name,
            dims=self.dims,
            coords=coords,
            attrs={
                "name": self.name,
                "units": self.unit.name,
                "eumType": self.type,
                "eumUnit": self.unit,
            },
        )
        return xr_da

    # ===============================================

    def __repr__(self) -> str:

        out = ["<mikeio.DataArray>"]
        if self.name is not None:
            out.append(f"name: {self.name}")

        rest = [
            self._dims_txt(),
            self._time_txt(),
            self._geometry_txt(),
            self._values_txt(),
        ]
        out = out + [x for x in rest if x is not None]

        return "\n".join(out)

    def _dims_txt(self) -> str:
        dims = [f"{self.dims[i]}:{self.shape[i]}" for i in range(self.ndim)]
        dimsstr = ", ".join(dims)
        return f"dims: ({dimsstr})"

    def _time_txt(self) -> str:
        noneq_txt = "" if self.is_equidistant else " non-equidistant"
        timetxt = (
            f"time: {self.time[0]} (time-invariant)"
            if self.n_timesteps == 1
            else f"time: {self.time[0]} - {self.time[-1]} ({self.n_timesteps}{noneq_txt} records)"
        )
        return timetxt

    def _geometry_txt(self) -> str:
        if not isinstance(self.geometry, (GeometryUndefined, type(None))):
            return f"geometry: {self.geometry}"

    def _values_txt(self) -> str:

        if self.ndim == 0 or (self.ndim == 1 and len(self.values) == 1):
            return f"values: {self.values}"
        elif self.ndim == 1 and len(self.values) < 5:
            valtxt = ", ".join([f"{v:0.4g}" for v in self.values])
            return f"values: [{valtxt}]"
        elif self.ndim == 1:
            return f"values: [{self.values[0]:0.4g}, {self.values[1]:0.4g}, ..., {self.values[-1]:0.4g}]"
