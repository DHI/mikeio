import warnings
from typing import Iterable, Sequence, Union, Mapping
import numpy as np
import pandas as pd
from copy import deepcopy
from mikeio.eum import EUMType, EUMUnit, ItemInfo

from .base import TimeSeries


class DataArray(TimeSeries):

    deletevalue = 1.0e-35

    def __init__(self, data, time: Union[pd.DatetimeIndex, str], item: ItemInfo = None):

        if not hasattr(data, "shape"):
            raise TypeError(
                "Data must be ArrayLike, e.g. numpy array, but it lacks a shape property"
            )
        if not hasattr(data, "ndim"):
            raise TypeError(
                "Data must be ArrayLike, e.g. numpy array, but it lacks a ndim property"
            )
        if not hasattr(data, "dtype"):
            raise TypeError(
                "Data must be ArrayLike, e.g. numpy array, it lacks a dtype property"
            )

        self.data: np.ndarray = data
        self.time = time
        self.item = item

    def __getitem__(self, key) -> "DataArray":
        da = self.copy()
        da.data = da.data[key]
        return da

    def plot(self, ax=None):
        import matplotlib.pyplot as plt

        if self.ndim != 1:
            raise NotImplementedError()

        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(self.time, self.data)
        ax.set_xlabel("time")
        ax.set_ylabel(f"{self.name} [{self.unit.name}]")

        return ax

    def to_numpy(self) -> np.ndarray:
        return self.data

    def flipud(self) -> "DataArray":
        """Flip updside down"""

        self.data = np.flip(self.data, axis=1)
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
            data = self.data + sign * other.data
        except:
            raise ValueError("Could not add data")

        new_da = self.copy()

        new_da.data = data

        return new_da

    def _add_value(self, value):
        try:
            data = value + self.data
        except:
            raise ValueError(f"{value} could not be added to DataArray")

        new_da = self.copy()

        new_da.data = data

        return new_da

    def _multiply_value(self, value):
        try:
            data = value * self.data
        except:
            raise ValueError(f"{value} could not be multiplied to DataArray")
        new_da = self.copy()

        new_da.data = data

        return new_da

    def copy(self):

        return deepcopy(self)

    @property
    def name(self) -> str:
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
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    def __repr__(self):

        out = ["<mikeio.DataArray>"]
        out.append(f"Name: {self.name}")
        out.append(f"Dimensions: {self.shape}")
        out.append(f"Time: {self.time[0]} - {self.time[-1]}")
        # if not self.is_equidistant:
        #    out.append("-- Non-equidistant calendar axis --")

        return str.join("\n", out)
