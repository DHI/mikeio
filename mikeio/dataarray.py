import warnings
from copy import deepcopy
from datetime import datetime
from functools import cached_property
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from mikecore.DfsuFile import DfsuFileType  # type: ignore

from .base import TimeSeries
from .data_utils import DataUtilsMixin
from .eum import EUMType, EUMUnit, ItemInfo
from .spatial.FM_geometry import (
    GeometryFM,
    GeometryFMAreaSpectrum,
    GeometryFMLineSpectrum,
    GeometryFMPointSpectrum,
    GeometryFMVerticalColumn,
    GeometryFMVerticalProfile,
    _GeometryFMLayered,
)
from .spatial.geometry import (
    GeometryPoint2D,
    GeometryPoint3D,
    GeometryUndefined,
)
from .spatial.grid_geometry import Grid1D, Grid2D, Grid3D
from .spectral import calc_m0_from_spectrum
from .data_plot import (
    _DataArrayPlotter,
    _DataArrayPlotterFM,
    _DataArrayPlotterGrid1D,
    _DataArrayPlotterGrid2D,
    _DataArrayPlotterAreaSpectrum,
    _DataArrayPlotterFMVerticalColumn,
    _DataArrayPlotterFMVerticalProfile,
    _DataArrayPlotterPointSpectrum,
    _DataArrayPlotterLineSpectrum,
)


class _DataArraySpectrumToHm0:
    def __init__(self, da: "DataArray") -> None:
        self.da = da

    def __call__(self, tail=True):
        # TODO: if action_density
        m0 = calc_m0_from_spectrum(
            self.da.to_numpy(),
            self.da.frequencies,
            self.da.directions,
            tail,
        )
        Hm0 = 4 * np.sqrt(m0)
        dims = tuple([d for d in self.da.dims if d not in ("frequency", "direction")])
        item = ItemInfo(EUMType.Significant_wave_height)
        g = self.da.geometry
        if isinstance(g, GeometryFMPointSpectrum):
            geometry = GeometryPoint2D(x=g.x, y=g.y)
        elif isinstance(g, GeometryFMLineSpectrum):
            geometry = Grid1D(
                nx=g.n_nodes,
                dx=1.0,
                node_coordinates=g.node_coordinates,
                axis_name="node",
            )
        elif isinstance(g, GeometryFMAreaSpectrum):
            geometry = GeometryFM(
                node_coordinates=g.node_coordinates,
                codes=g.codes,
                node_ids=g.node_ids,
                projection=g.projection_string,
                element_table=g.element_table,
                element_ids=g.element_ids,
            )
        else:
            geometry = GeometryUndefined()

        return DataArray(
            data=Hm0, time=self.da.time, item=item, dims=dims, geometry=geometry
        )


class DataArray(DataUtilsMixin, TimeSeries):
    """DataArray with data and metadata for a single item in a dfs file

    The DataArray has these main properties:

    * time - a pandas.DatetimeIndex with the time instances of the data
    * geometry - a geometry object e.g. Grid2D or GeometryFM
    * values - a numpy array containing the data
    * item - an ItemInfo with name, type and unit
    """

    deletevalue = 1.0e-35

    def __init__(
        self,
        data,
        *,
        time: Optional[Union[pd.DatetimeIndex, str]] = None,
        item: Optional[ItemInfo] = None,
        geometry=GeometryUndefined(),
        zn=None,
        dims: Optional[Sequence[str]] = None,
    ):
        # TODO: add optional validation validate=True
        self._values = self._parse_data(data)
        self.time: pd.DatetimeIndex = self._parse_time(time)
        self.dims = self._parse_dims(dims, geometry)

        self._check_time_data_length(self.time)

        self.item = self._parse_item(item)
        self.geometry = self._parse_geometry(geometry, self.dims, self.shape)
        self._zn = self._parse_zn(zn, self.geometry, self.n_timesteps)
        self._set_spectral_attributes(geometry)
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
            if ndim_no_time == 1:
                dims.append("frequency")
            if ndim_no_time == 2:
                dims.append("direction")
                dims.append("frequency")
        elif isinstance(geometry, GeometryFM):
            if geometry._type == DfsuFileType.DfsuSpectral1D:
                if ndim_no_time > 0:
                    dims.append("node")
            else:
                if ndim_no_time > 0:
                    dims.append("element")
            if geometry.is_spectral:
                if ndim_no_time == 2:
                    dims.append("frequency")
                elif ndim_no_time == 3:
                    dims.append("direction")
                    dims.append("frequency")
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
            if dims == ("time", "x"):
                return Grid1D(nx=shape[1], dx=1.0 / (shape[1] - 1))

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
                shape[axis] == geometry.nx
            ), "data shape does not match number of grid points"
        elif isinstance(geometry, Grid2D):
            assert shape[axis] == geometry.ny, "data shape does not match ny"
            assert shape[axis + 1] == geometry.nx, "data shape does not match nx"
        # elif isinstance(geometry, Grid3D): # TODO

        return geometry

    @staticmethod
    def _parse_zn(zn, geometry, n_timesteps):
        if zn is not None:
            if isinstance(geometry, _GeometryFMLayered):
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
        elif isinstance(self.geometry, GeometryFMPointSpectrum):
            return _DataArrayPlotterPointSpectrum(self)
        elif isinstance(self.geometry, GeometryFMLineSpectrum):
            return _DataArrayPlotterLineSpectrum(self)
        elif isinstance(self.geometry, GeometryFMAreaSpectrum):
            return _DataArrayPlotterAreaSpectrum(self)
        elif isinstance(self.geometry, GeometryFM):
            return _DataArrayPlotterFM(self)
        elif isinstance(self.geometry, Grid1D):
            return _DataArrayPlotterGrid1D(self)
        elif isinstance(self.geometry, Grid2D):
            return _DataArrayPlotterGrid2D(self)
        else:
            return _DataArrayPlotter(self)

    def _set_spectral_attributes(self, geometry):
        if hasattr(geometry, "frequencies") and hasattr(geometry, "directions"):
            self.frequencies = geometry.frequencies
            self.n_frequencies = geometry.n_frequencies
            self.directions = geometry.directions
            self.n_directions = geometry.n_directions
            self.to_Hm0 = _DataArraySpectrumToHm0(self)

    # ============= Basic properties/methods ===========

    @property
    def name(self) -> Optional[str]:
        """Name of this DataArray (=da.item.name)"""
        return self.item.name

    @name.setter
    def name(self, value):
        self.item.name = value

    @property
    def type(self) -> EUMType:
        """EUMType"""
        return self.item.type

    @property
    def unit(self) -> EUMUnit:
        """EUMUnit"""
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

    @cached_property
    def is_equidistant(self) -> bool:
        """Is DataArray equidistant in time?"""
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
            first: pd.Timestamp = self.time[0]  # type: ignore
            second: pd.Timestamp = self.time[1]  # type: ignore
            dt = (second - first).total_seconds()
        return dt

    @property
    def n_timesteps(self) -> int:
        """Number of time steps"""
        return len(self.time)

    @property
    def shape(self):
        """Tuple of array dimensions"""
        return self.values.shape

    @property
    def ndim(self) -> int:
        """Number of array dimensions"""
        return self.values.ndim

    @property
    def dtype(self):
        """Data-type of the array elements"""
        return self.values.dtype

    @property
    def values(self) -> np.ndarray:
        """Values as a np.ndarray (equivalent to to_numpy())"""
        return self._values

    @values.setter
    def values(self, value):
        if np.isscalar(self._values):
            if not np.isscalar(value):
                raise ValueError("Shape of new data is wrong (should be scalar)")
        elif value.shape != self._values.shape:
            raise ValueError("Shape of new data is wrong")

        self._values = value

    def to_numpy(self) -> np.ndarray:
        """Values as a np.ndarray (equivalent to values)"""
        return self._values

    @property
    def _has_time_axis(self):
        return self.dims[0][0] == "t"

    def dropna(self) -> "DataArray":
        """Remove time steps where values are NaN"""
        if not self._has_time_axis:
            raise ValueError("Not available if no time axis!")

        x = self.to_numpy()

        # this seems overly complicated...
        axes = tuple(range(1, x.ndim))
        idx = list(np.where(~np.isnan(x).all(axis=axes))[0])
        return self.isel(idx, axis=0)

    def flipud(self) -> "DataArray":
        """Flip upside down (on first non-time axis)"""

        first_non_t_axis = 1 if self._has_time_axis else 0
        self.values = np.flip(self.values, axis=first_non_t_axis)
        return self

    def describe(self, **kwargs) -> pd.DataFrame:
        """Generate descriptive statistics by wrapping :py:meth:`pandas.DataFrame.describe`"""
        data = {}
        data[self.name] = self.to_numpy().ravel()
        df = pd.DataFrame(data).describe(**kwargs)

        return df

    def copy(self) -> "DataArray":
        """Make copy of DataArray"""
        return deepcopy(self)

    def squeeze(self) -> "DataArray":
        """Remove axes of length 1

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

    # TODO implement def where() modelled after xarray
    # def get_masked(self, key) -> np.ndarray:
    #    if self._is_boolean_mask(key):
    #        mask = key if isinstance(key, np.ndarray) else key.values
    #        return self._get_by_boolean_mask(self.values, mask)
    #    else:
    #        raise ValueError("Invalid mask")

    def __getitem__(self, key) -> "DataArray":

        da = self
        dims = self.dims
        key = self._getitem_parse_key(key)
        for j, k in enumerate(key):
            if isinstance(k, Iterable) or k != slice(None):
                if dims[j] == "time":
                    # getitem accepts fancy indexing only for time
                    k = self._get_time_idx_list(self.time, k)
                    if self._n_selected_timesteps(self.time, k) == 0:
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
            if any_idx_after_0_time and self._has_time_axis:
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
        """Return a new DataArray whose data is given by
        integer indexing along the specified dimension(s).

        Note that the data will be a _view_ of the original data
        if possible (single index or slice), otherwise a copy (fancy indexing)
        following NumPy convention.

        The spatial parameters available depend on the dims
        (i.e. geometry) of the DataArray:

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
        DataArray
            new DataArray with selected data

        See Also
        --------
        dims : Get axis names
        sel : Select data using labels

        Examples
        --------
        >>> da = mikeio.read("europe_wind_long_lat.dfs2")[0]
        >>> da
        <mikeio.DataArray>
        name: Mean Sea Level Pressure
        dims: (time:1, y:101, x:221)
        time: 2012-01-01 00:00:00 (time-invariant)
        geometry: Grid2D (ny=101, nx=221)
        >>> da.isel(time=-1)
        <mikeio.DataArray>
        name: Mean Sea Level Pressure
        dims: (y:101, x:221)
        time: 2012-01-01 00:00:00 (time-invariant)
        geometry: Grid2D (ny=101, nx=221)
        >>> da.isel(x=slice(10,20), y=slice(40,60))
        <mikeio.DataArray>
        name: Mean Sea Level Pressure
        dims: (time:1, y:20, x:10)
        time: 2012-01-01 00:00:00 (time-invariant)
        geometry: Grid2D (ny=20, nx=10)
        >>> da.isel(y=34)
        <mikeio.DataArray>
        name: Mean Sea Level Pressure
        dims: (time:1, x:221)
        time: 2012-01-01 00:00:00 (time-invariant)
        geometry: Grid1D (n=221, dx=0.25)

        >>> da = mikeio.read("oresund_sigma_z.dfsu").Temperature
        >>> da
        <mikeio.DataArray>
        name: Temperature
        dims: (time:3, element:17118)
        time: 1997-09-15 21:00:00 - 1997-09-16 03:00:00 (3 records)
        geometry: Dfsu3DSigmaZ (17118 elements, 4 sigma-layers, 5 z-layers)
        >>> da.isel(element=45)
        <mikeio.DataArray>
        name: Temperature
        dims: (time:3)
        time: 1997-09-15 21:00:00 - 1997-09-16 03:00:00 (3 records)
        geometry: GeometryPoint3D(x=328717.05429134873, y=6143529.158495431, z=-4.0990404685338335)
        values: [17.29, 17.25, 17.19]
        >>> da.isel(element=range(200))
        <mikeio.DataArray>
        name: Temperature
        dims: (time:3, element:200)
        time: 1997-09-15 21:00:00 - 1997-09-16 03:00:00 (3 records)
        geometry: Dfsu3DSigmaZ (200 elements, 3 sigma-layers, 3 z-layers)
        """
        if isinstance(self.geometry, Grid2D) and ("x" in kwargs and "y" in kwargs):
            idx_x = kwargs["x"]
            idx_y = kwargs["y"]
            return self.isel(x=idx_x).isel(y=idx_y)
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

        idx_slice = None
        if isinstance(idx, slice):
            idx_slice = idx
            idx = list(range(*idx.indices(self.shape[axis])))
        if idx is None or (not np.isscalar(idx) and len(idx) == 0):
            raise ValueError(
                "Empty index is not allowed"
            )  # TODO other option would be to have a NullDataArray

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

            # TOOD this is ugly
            if isinstance(geometry, _GeometryFMLayered):
                node_ids, _ = self.geometry._get_nodes_and_table_for_elements(
                    idx, node_layers="all"
                )
                zn = self._zn[:, node_ids]

        # reduce dims only if singleton idx
        dims = (
            tuple([d for i, d in enumerate(self.dims) if i != axis])
            if single_index
            else self.dims
        )
        if single_index:
            idx = int(idx)
        elif idx_slice is not None:
            idx = idx_slice

        if axis == 0:
            dat = self.values[idx]
        elif axis == 1:
            dat = self.values[:, idx]
        elif axis == 2:
            dat = self.values[:, :, idx]
        elif axis == 3:
            dat = self.values[:, :, :, idx]
        else:
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
        time: Optional[Union[str, pd.DatetimeIndex, "DataArray"]] = None,
        **kwargs,
    ) -> "DataArray":
        """Return a new DataArray whose data is given by
        selecting index labels along the specified dimension(s).

        In contrast to DataArray.isel, indexers for this method
        should use labels instead of integers.

        The spatial parameters available depend on the geometry of the DataArray:

        * Grid1D: x
        * Grid2D: x, y, coords, area
        * Grid3D: [not yet implemented! use isel instead]
        * GeometryFM: (x,y), coords, area
        * GeometryFMLayered: (x,y,z), coords, area, layers

        Parameters
        ----------
        time : Union[str, pd.DatetimeIndex, DataArray], optional
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
        DataArray
            new DataArray with selected data

        See Also
        --------
        isel : Select data using integer indexing
        interp : Interp data in time and space

        Examples
        --------
        >>> da = mikeio.read("random.dfs1")[0]
        >>> da
        <mikeio.DataArray>
        name: testing water level
        dims: (time:100, x:3)
        time: 2012-01-01 00:00:00 - 2012-01-01 00:19:48 (100 records)
        geometry: Grid1D (n=3, dx=100)
        >>> da.sel(time=slice(None, "2012-1-1 00:02"))
        <mikeio.DataArray>
        name: testing water level
        dims: (time:15, x:3)
        time: 2012-01-01 00:00:00 - 2012-01-01 00:02:48 (15 records)
        geometry: Grid1D (n=3, dx=100)
        >>> da.sel(x=100)
        <mikeio.DataArray>
        name: testing water level
        dims: (time:100)
        time: 2012-01-01 00:00:00 - 2012-01-01 00:19:48 (100 records)
        values: [0.3231, 0.6315, ..., 0.7506]

        >>> da = mikeio.read("oresund_sigma_z.dfsu").Temperature
        >>> da
        <mikeio.DataArray>
        name: Temperature
        dims: (time:3, element:17118)
        time: 1997-09-15 21:00:00 - 1997-09-16 03:00:00 (3 records)
        geometry: Dfsu3DSigmaZ (17118 elements, 4 sigma-layers, 5 z-layers)
        >>> da.sel(time="1997-09-15")
        <mikeio.DataArray>
        name: Temperature
        dims: (element:17118)
        time: 1997-09-15 21:00:00 (time-invariant)
        geometry: Dfsu3DSigmaZ (17118 elements, 4 sigma-layers, 5 z-layers)
        values: [16.31, 16.43, ..., 16.69]
        >>> da.sel(x=340000, y=6160000, z=-3)
        <mikeio.DataArray>
        name: Temperature
        dims: (time:3)
        time: 1997-09-15 21:00:00 - 1997-09-16 03:00:00 (3 records)
        geometry: GeometryPoint3D(x=340028.1116933554, y=6159980.070243686, z=-3.0)
        values: [17.54, 17.31, 17.08]
        >>> da.sel(area=(340000, 6160000, 350000, 6170000))
        <mikeio.DataArray>
        name: Temperature
        dims: (time:3, element:224)
        time: 1997-09-15 21:00:00 - 1997-09-16 03:00:00 (3 records)
        geometry: Dfsu3DSigmaZ (224 elements, 3 sigma-layers, 1 z-layers)
        >>> da.sel(layers="bottom")
        <mikeio.DataArray>
        name: Temperature
        dims: (time:3, element:3700)
        time: 1997-09-15 21:00:00 - 1997-09-16 03:00:00 (3 records)
        geometry: Dfsu2D (3700 elements, 2090 nodes)
        """
        da = self

        # select in space
        if len(kwargs) > 0:
            idx = self.geometry.find_index(**kwargs)
            if isinstance(idx, tuple):
                # TODO: support for dfs3
                assert len(idx) == 2
                t_ax_offset = 1 if self._has_time_axis else 0
                ii, jj = idx
                if jj is not None:
                    da = da.isel(idx=jj, axis=(0 + t_ax_offset))
                if ii is not None:
                    sp_axis = 0 if (jj is not None and len(jj) == 1) else 1
                    da = da.isel(idx=ii, axis=(sp_axis + t_ax_offset))
            else:
                da = da.isel(idx, axis="space")

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
        # *, # TODO: make this a keyword-only argument in the future
        time: Optional[Union[pd.DatetimeIndex, "DataArray"]] = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        n_nearest: int = 3,
        interpolant=None,
        **kwargs,
    ) -> "DataArray":
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
        time : Union[float, pd.DatetimeIndex, DataArray], optional
            timestep in seconds or discrete time instances given by
            pd.DatetimeIndex (typically from another DataArray
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
        DataArray
            new DataArray with interped data

        See Also
        --------
        sel : Select data using label indexing
        interp_like : Interp to another time/space of another DataArray
        interp_time : Interp in the time direction only

        Examples
        --------
        >>> da = mikeio.read("random.dfs1")[0]
        >>> da.interp(time=3600)
        >>> da.interp(x=110)

        >>> da = mikeio.read("HD2D.dfsu").Salinity
        >>> da.interp(x=340000, y=6160000)
        """
        if z is not None:
            raise NotImplementedError()

        geometry: Union[
            GeometryPoint2D, GeometryPoint3D, GeometryUndefined
        ] = GeometryUndefined()

        # interp in space
        if (x is not None) or (y is not None) or (z is not None):
            coords = [(x, y)]

            if isinstance(self.geometry, Grid2D):  # TODO DIY bilinear interpolation
                if x is None or y is None:
                    raise ValueError("both x and y must be specified")

                xr_da = self.to_xarray()
                dai = xr_da.interp(x=x, y=y).values
                geometry = GeometryPoint2D(
                    x=x, y=y, projection=self.geometry.projection
                )
            elif isinstance(self.geometry, Grid1D):
                if interpolant is None:
                    interpolant = self.geometry.get_spatial_interpolant(coords)
                dai = self.geometry.interp(self.to_numpy(), *interpolant).flatten()
                geometry = GeometryUndefined()
            elif isinstance(self.geometry, GeometryFM):
                if x is None or y is None:
                    raise ValueError("both x and y must be specified")
                if self.geometry.is_layered:
                    raise NotImplementedError(
                        "Interpolation in 3d is not yet implemented"
                    )

                if interpolant is None:
                    interpolant = self.geometry.get_2d_interpolant(
                        coords, n_nearest=n_nearest, **kwargs
                    )
                dai = self.geometry.interp2d(self, *interpolant).flatten()
                if z is None:
                    geometry = GeometryPoint2D(
                        x=x, y=y, projection=self.geometry.projection
                    )
                else:
                    geometry = GeometryPoint3D(
                        x=x, y=y, z=z, projection=self.geometry.projection
                    )

            da = DataArray(
                data=dai, time=self.time, geometry=geometry, item=deepcopy(self.item)
            )
        else:
            da = self.copy()

        # interp in time
        if time is not None:
            da = da.interp_time(time)

        return da

    def __dataarray_read_item_time_func(
        self, item: int, step: int
    ) -> Tuple[np.ndarray, float]:
        "Used by _extract_track"
        # Ignore item argument
        data = self.isel(time=step).to_numpy()
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

        return _extract_track(
            deletevalue=self.deletevalue,
            start_time=self.start_time,
            end_time=self.end_time,
            timestep=self.timestep,
            geometry=self.geometry,
            n_elements=self.shape[1],  # TODO is there a better way to find out this?
            track=track,
            items=[self.item],
            time_steps=list(range(self.n_timesteps)),
            item_numbers=[0],
            method=method,
            dtype=dtype,
            data_read_func=lambda item, step: self.__dataarray_read_item_time_func(
                item, step
            ),
        )

    def interp_time(
        self,
        dt: Union[float, pd.DatetimeIndex, "DataArray"],
        *,
        method="linear",
        extrapolate=True,
        fill_value=np.nan,
    ) -> "DataArray":
        """Temporal interpolation

        Wrapper of :py:class:`scipy.interpolate.interp1d`

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
            data=data,
            time=t_out_index,
            item=deepcopy(self.item),
            geometry=self.geometry,
            zn=zn,
        )

    def interp_na(self, axis="time", **kwargs) -> "DataArray":
        """Fill in NaNs by interpolating according to different methods.

        Wrapper of :py:meth:`xarray.DataArray.interpolate_na`

        Examples
        --------

        >>> time = pd.date_range("2000", periods=3, freq="D")
        >>> da = mikeio.DataArray(data=np.array([0.0, np.nan, 2.0]), time=time)
        >>> da
        <mikeio.DataArray>
        name: NoName
        dims: (time:3)
        time: 2000-01-01 00:00:00 - 2000-01-03 00:00:00 (3 records)
        values: [0, nan, 2]
        >>> da.interp_na()
        <mikeio.DataArray>
        name: NoName
        dims: (time:3)
        time: 2000-01-01 00:00:00 - 2000-01-03 00:00:00 (3 records)
        values: [0, 1, 2]
        """

        xr_da = self.to_xarray().interpolate_na(dim=axis, **kwargs)
        self.values = xr_da.values
        return self

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
        if not (isinstance(self.geometry, GeometryFM) and self.geometry.is_2d):
            raise NotImplementedError(
                "Currently only supports interpolating from 2d flexible mesh data!"
            )

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
            elem_ids, weights = self.geometry.get_2d_interpolant(xy, **kwargs)
        else:
            elem_ids, weights = interpolant

        if isinstance(geom, (Grid2D, GeometryFM)):
            shape = (geom.ny, geom.nx) if isinstance(geom, Grid2D) else None

            dai = self.geometry.interp2d(
                data=self.to_numpy(), elem_ids=elem_ids, weights=weights, shape=shape
            )

        dai = DataArray(
            data=dai, time=self.time, geometry=geom, item=deepcopy(self.item)
        )

        if hasattr(other, "time"):
            dai = dai.interp_time(other.time)

        return dai

    @staticmethod
    def concat(dataarrays: Sequence["DataArray"], keep="last") -> "DataArray":
        """Concatenate DataArrays along the time axis

        Parameters
        ---------
        dataarrays: sequence of DataArrays
        keep: str, optional
            TODO Yet to be implemented, default: last

        Returns
        -------
        DataArray
            The concatenated DataArray

        Examples
        --------
        >>> import mikeio
        >>> da1 = mikeio.read("HD2D.dfsu", time=[0,1])[0]
        >>> da2 = mikeio.read("HD2D.dfsu", time=[2,3])[0]
        >>> da1.n_timesteps
        2
        >>> da3 = DataArray.concat([da1,da2])
        >>> da3.n_timesteps
        4
        """
        from mikeio import Dataset

        datasets = [Dataset([da]) for da in dataarrays]

        ds = Dataset.concat(datasets, keep=keep)
        da = ds[0]
        assert isinstance(da, DataArray)
        return da

    # ============= Aggregation methods ===========

    def max(self, axis=0, **kwargs) -> "DataArray":
        """Max value along an axis

        Parameters
        ----------
        axis: (int, str, None), optional
            axis number or "time" or "space", by default 0

        Returns
        -------
        DataArray
            array with max values

        See Also
        --------
            nanmax : Max values with NaN values removed
        """
        return self.aggregate(axis=axis, func=np.max, **kwargs)

    def min(self, axis=0, **kwargs) -> "DataArray":
        """Min value along an axis

        Parameters
        ----------
        axis: (int, str, None), optional
            axis number or "time" or "space", by default 0

        Returns
        -------
        DataArray
            array with min values

        See Also
        --------
            nanmin : Min values with NaN values removed
        """
        return self.aggregate(axis=axis, func=np.min, **kwargs)

    def mean(self, axis=0, **kwargs) -> "DataArray":
        """Mean value along an axis

        Parameters
        ----------
        axis: (int, str, None), optional
            axis number or "time" or "space", by default 0

        Returns
        -------
        DataArray
            array with mean values

        See Also
        --------
            nanmean : Mean values with NaN values removed
        """
        return self.aggregate(axis=axis, func=np.mean, **kwargs)

    def std(self, axis=0, **kwargs) -> "DataArray":
        """Standard deviation values along an axis

        Parameters
        ----------
        axis: (int, str, None), optional
            axis number or "time" or "space", by default 0

        Returns
        -------
        DataArray
            array with standard deviation values

        See Also
        --------
            nanstd : Standard deviation values with NaN values removed
        """
        return self.aggregate(axis=axis, func=np.std, **kwargs)

    def ptp(self, axis=0, **kwargs) -> "DataArray":
        """Range (max - min) a.k.a Peak to Peak along an axis

        Parameters
        ----------
        axis: (int, str, None), optional
            axis number or "time" or "space", by default 0

        Returns
        -------
        DataArray
            array with peak to peak values
        """
        return self.aggregate(axis=axis, func=np.ptp, **kwargs)

    def average(self, weights, axis=0, **kwargs) -> "DataArray":
        """Compute the weighted average along the specified axis.

        Parameters
        ----------
        axis: (int, str, None), optional
            axis number or "time" or "space", by default 0

        Returns
        -------
        DataArray
            DataArray with weighted average values

        See Also
        --------
            aggregate : Weighted average

        Examples
        --------
        >>> dfs = Dfsu("HD2D.dfsu")
        >>> da = dfs.read(["Current speed"])[0]
        >>> area = dfs.get_element_area()
        >>> da2 = da.average(axis="space", weights=area)
        """

        def func(x, axis, keepdims):
            if keepdims:
                raise NotImplementedError()

            return np.average(x, weights=weights, axis=axis)

        return self.aggregate(axis=axis, func=func, **kwargs)

    def nanmax(self, axis=0, **kwargs) -> "DataArray":
        """Max value along an axis (NaN removed)

        Parameters
        ----------
        axis: (int, str, None), optional
            axis number or "time" or "space", by default 0

        Returns
        -------
        DataArray
            array with max values

        See Also
        --------
            nanmax : Max values with NaN values removed
        """
        return self.aggregate(axis=axis, func=np.nanmax, **kwargs)

    def nanmin(self, axis=0, **kwargs) -> "DataArray":
        """Min value along an axis (NaN removed)

        Parameters
        ----------
        axis: (int, str, None), optional
            axis number or "time" or "space", by default 0

        Returns
        -------
        DataArray
            array with min values

        See Also
        --------
            nanmin : Min values with NaN values removed
        """
        return self.aggregate(axis=axis, func=np.nanmin, **kwargs)

    def nanmean(self, axis=0, **kwargs) -> "DataArray":
        """Mean value along an axis (NaN removed)

        Parameters
        ----------
        axis: (int, str, None), optional
            axis number or "time" or "space", by default 0

        Returns
        -------
        DataArray
            array with mean values

        See Also
        --------
            mean : Mean values
        """
        return self.aggregate(axis=axis, func=np.nanmean, **kwargs)

    def nanstd(self, axis=0, **kwargs) -> "DataArray":
        """Standard deviation value along an axis (NaN removed)

        Parameters
        ----------
        axis: (int, str, None), optional
            axis number or "time" or "space", by default 0

        Returns
        -------
        DataArray
            array with standard deviation values

        See Also
        --------
            std : Standard deviation
        """
        return self.aggregate(axis=axis, func=np.nanstd, **kwargs)

    def aggregate(self, axis=0, func=np.nanmean, **kwargs) -> "DataArray":
        """Aggregate along an axis

        Parameters
        ----------
        axis: (int, str, None), optional
            axis number or "time" or "space", by default 0
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

        item = deepcopy(self.item)
        if "name" in kwargs:
            item.name = kwargs.pop("name")

        with warnings.catch_warnings():  # there might be all-Nan slices, it is ok, so we ignore them!
            warnings.simplefilter("ignore", category=RuntimeWarning)
            data = func(self.to_numpy(), axis=axis, keepdims=False, **kwargs)

        if axis == 0 and "time" in self.dims:  # time
            geometry = self.geometry
            zn = None if self._zn is None else self._zn[0]

        else:
            geometry = GeometryUndefined()
            zn = None

        return DataArray(
            data=data,
            time=time,
            item=item,
            geometry=geometry,
            dims=dims,
            zn=zn,
        )

    def quantile(self, q, *, axis=0, **kwargs):
        """Compute the q-th quantile of the data along the specified axis.

        Wrapping np.quantile

        Parameters
        ----------
        q: array_like of float
            Quantile or sequence of quantiles to compute,
            which must be between 0 and 1 inclusive.
        axis: (int, str, None), optional
            axis number or "time" or "space", by default 0

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

    def nanquantile(self, q, *, axis=0, **kwargs):
        """Compute the q-th quantile of the data along the specified axis, while ignoring nan values.

        Wrapping np.nanquantile

        Parameters
        ----------
        q: array_like of float
            Quantile or sequence of quantiles to compute,
            which must be between 0 and 1 inclusive.
        axis: (int, str, None), optional
            axis number or "time" or "space", by default 0

        Returns
        -------
        DataArray
            data with quantile values

        Examples
        --------
        >>> da.nanquantile(q=[0.25,0.75])
        >>> da.nanquantile(q=0.5)
        >>> da.nanquantile(q=[0.01,0.5,0.99], axis="space")

        See Also
        --------
        quantile : Quantile with NaN values
        """
        return self._quantile(q, axis=axis, func=np.nanquantile, **kwargs)

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
            return DataArray(
                data=qdat, time=time, item=item, geometry=geometry, dims=dims, zn=zn
            )
        else:
            res = []
            for quantile in q:
                qd = self._quantile(q=quantile, axis=axis, func=func)
                newname = f"Quantile {quantile}, {self.name}"
                qd.name = newname
                res.append(qd)

            return Dataset(data=res, validate=False)

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
        return self._apply_math_operation(other, np.multiply, "x")  # x in place of *

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

    def __eq__(self, other) -> "DataArray":  # type: ignore
        bmask = self.values == self._other_to_values(other)
        return self._boolmask_to_new_DataArray(bmask)

    def __ne__(self, other) -> "DataArray":  # type: ignore
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

    def to_dfs(self, filename, **kwargs) -> None:
        """Write data to a new dfs file

        Parameters
        ----------
        filename: str
            full path to the new dfs file
        dtype: str, np.dtype, DfsSimpleType, optional
            Dfs0 only: set the dfs data type of the written data
            to e.g. np.float64, by default: DfsSimpleType.Float (=np.float32)
        """
        self._to_dataset().to_dfs(filename, **kwargs)

    def to_xarray(self):
        """Export to xarray.DataArray"""

        import xarray as xr

        coords = {}
        if self._has_time_axis:
            coords["time"] = xr.DataArray(self.time, dims="time")

        if isinstance(self.geometry, Grid1D):
            coords["x"] = xr.DataArray(data=self.geometry.x, dims="x")
        elif isinstance(self.geometry, Grid2D):
            coords["y"] = xr.DataArray(data=self.geometry.y, dims="y")
            coords["x"] = xr.DataArray(data=self.geometry.x, dims="x")
        elif isinstance(self.geometry, Grid3D):
            coords["z"] = xr.DataArray(data=self.geometry.z, dims="z")
            coords["y"] = xr.DataArray(data=self.geometry.y, dims="y")
            coords["x"] = xr.DataArray(data=self.geometry.x, dims="x")
        elif isinstance(self.geometry, GeometryFM):
            coords["element"] = xr.DataArray(
                data=self.geometry.element_ids, dims="element"
            )
        elif isinstance(self.geometry, GeometryPoint2D):
            coords["x"] = self.geometry.x
            coords["y"] = self.geometry.y
        elif isinstance(self.geometry, GeometryPoint3D):
            coords["x"] = self.geometry.x
            coords["y"] = self.geometry.y
            coords["z"] = self.geometry.z

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
            f"time: {str(self.time[0])} (time-invariant)"
            if self.n_timesteps == 1
            else f"time: {str(self.time[0])} - {str(self.time[-1])} ({self.n_timesteps}{noneq_txt} records)"
        )
        return timetxt

    def _geometry_txt(self) -> str:
        return f"geometry: {self.geometry}"

    def _values_txt(self) -> str:

        if self.ndim == 0 or (self.ndim == 1 and len(self.values) == 1):
            return f"values: {self.values}"
        elif self.ndim == 1 and len(self.values) < 5:
            valtxt = ", ".join([f"{v:0.4g}" for v in self.values])
            return f"values: [{valtxt}]"
        elif self.ndim == 1:
            return f"values: [{self.values[0]:0.4g}, {self.values[1]:0.4g}, ..., {self.values[-1]:0.4g}]"
        else:
            return ""  # raise NotImplementedError()
