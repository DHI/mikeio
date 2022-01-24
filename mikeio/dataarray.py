import warnings
from typing import Iterable, Optional, Sequence, Tuple, Union, Mapping
import numpy as np
import pandas as pd
from copy import deepcopy
from mikeio.eum import EUMType, EUMUnit, ItemInfo

from .spatial.FM_geometry import GeometryFM
from .spatial.FM_utils import _plot_map

from .base import TimeSeries
from .spatial.geometry import _Geometry


class DataArray(TimeSeries):

    deletevalue = 1.0e-35

    def __init__(
        self,
        data,
        time: Union[pd.DatetimeIndex, str],
        item: ItemInfo = None,
        geometry: _Geometry = None,
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

        self._values = data
        self.time = time
        if (item is not None) and (not isinstance(item, ItemInfo)):
            raise ValueError("Item must be an ItemInfo")
        self.item = item

        self.geometry = geometry

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, value):
        if value.shape != self._values.shape:
            raise ValueError("Shape of new data is wrong")

        self._values = value

    def __getitem__(self, key) -> "DataArray":
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
        if isinstance(key, tuple):
            if isinstance(self.geometry, GeometryFM):
                # TODO: allow for selection of layers
                elements = key[1]
                if isinstance(elements, slice):
                    elements = list(range(*elements.indices(self.geometry.n_elements)))
                else:
                    elements = np.atleast_1d(elements)
                geometry = self.geometry.elements_to_geometry(elements)
                key = (steps, elements)
            else:
                # TODO: better handling of dfs1,2,3
                key = (steps, *key[1:])
        else:
            key = steps

        data = self._values[key].copy()
        return DataArray(data=data, time=time, item=self.item, geometry=geometry)

    def plot(self, ax=None, figsize=None, **kwargs):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        if self.ndim == 0:
            ax.plot(self.time, self.values, **kwargs)
            ax.set_xlabel("time")
            ax.set_ylabel(f"{self.name} [{self.unit.name}]")
            return ax

        if isinstance(self.geometry, GeometryFM):
            if self.geometry.is_2d:
                self._plot_FM_map(ax, **kwargs)
                return ax

        if self.ndim == 2:
            ax.imshow(self.values, **kwargs)
            return ax

        # if everything else fails, plot histogram
        ax.hist(self.values, **kwargs)
        return ax

    def _plot_FM_map(self, ax, **kwargs):
        if self.n_timesteps > 1:
            z = self.values[0]  # select first step as default plotting behaviour
        else:
            z = np.squeeze(self.values)

        if "label" not in kwargs:
            kwargs["label"] = f"{self.name} [{self.unit.name}]"
        if "title" not in kwargs:
            kwargs["title"] = f"{self.time[0]}"

        _plot_map(
            node_coordinates=self.geometry.node_coordinates,
            element_table=self.geometry.element_table,
            element_coordinates=self.geometry.element_coordinates,
            boundary_polylines=self.geometry.boundary_polylines,
            is_geo=self.geometry.is_geo,
            z=z,
            ax=ax,
            **kwargs,
        )

    def to_numpy(self) -> np.ndarray:
        return self._values

    def flipud(self) -> "DataArray":
        """Flip upside down"""

        self.values = np.flip(self.values, axis=1)
        return self

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
        out.append(f"Dimensions: {self.shape}")
        out.append(f"Time: {self.time[0]} - {self.time[-1]}")
        if not self.is_equidistant:
            out.append("-- Non-equidistant calendar axis --")

        return str.join("\n", out)
