from __future__ import annotations
import re
from typing import Iterable, Sequence, Sized, Tuple, Union, List

import numpy as np
import pandas as pd

from .._time import DateTimeSelector


def _to_safe_name(name: str) -> str:
    tmp = re.sub("[^0-9a-zA-Z]", "_", name)
    return re.sub("_+", "_", tmp)  # Collapse multiple underscores


def _n_selected_timesteps(x: Sized, k: slice | Sized) -> int:
    if isinstance(k, slice):
        k = list(range(*k.indices(len(x))))
    return len(k)


def _get_time_idx_list(time: pd.DatetimeIndex, steps) -> Union [List[int], slice]:
    """Find list of idx in DatetimeIndex"""

    # indexing with a slice needs to be handled differently, since slicing returns a view

    if isinstance(steps, slice):
        if isinstance(steps.start, int) and isinstance(steps.stop, int):
            return steps

    dts = DateTimeSelector(time)
    return dts.isel(steps)

# TODO this only used by DataArray, so consider to move it there
class DataUtilsMixin:
    """DataArray Utils"""

    @staticmethod
    def _to_safe_name(name: str) -> str:
        return _to_safe_name(name)

    @staticmethod
    def _time_by_agg_axis(
        time: pd.DatetimeIndex, axis: int | Sequence[int]
    ) -> pd.DatetimeIndex:
        """New DatetimeIndex after aggregating over time axis"""
        if axis == 0 or (isinstance(axis, Sequence) and 0 in axis):
            time = pd.DatetimeIndex([time[0]])

        return time

    @staticmethod
    def _get_time_idx_list(time: pd.DatetimeIndex, steps):
        """Find list of idx in DatetimeIndex"""

        return _get_time_idx_list(time, steps)

    @staticmethod
    def _n_selected_timesteps(time, k):
        return _n_selected_timesteps(time, k)

    @staticmethod
    def _is_boolean_mask(x) -> bool:
        if hasattr(x, "dtype"):  # isinstance(x, (np.ndarray, DataArray)):
            return x.dtype == np.dtype("bool")
        return False

    @staticmethod
    def _get_by_boolean_mask(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if data.shape != mask.shape:
            return data[np.broadcast_to(mask, data.shape)]
        return data[mask]

    @staticmethod
    def _set_by_boolean_mask(data: np.ndarray, mask: np.ndarray, value) -> None:
        if data.shape != mask.shape:
            data[np.broadcast_to(mask, data.shape)] = value
        else:
            data[mask] = value

    @staticmethod
    def _parse_time(time) -> pd.DatetimeIndex:
        """Allow anything that we can create a DatetimeIndex from"""
        if time is None:
            time = [pd.Timestamp(2018, 1, 1)] # TODO is this the correct epoch?
        if isinstance(time, str) or (not isinstance(time, Iterable)):
            time = [time]

        if not isinstance(time, pd.DatetimeIndex):
            index = pd.DatetimeIndex(time)
        else:
            index = time

        if not index.is_monotonic_increasing:
            raise ValueError(
                "Time must be monotonic increasing (only equal or increasing) instances."
            )
        assert isinstance(index, pd.DatetimeIndex)
        return index

    @staticmethod
    def _parse_axis(data_shape, dims, axis) -> int | Tuple[int]:
        # TODO change to return tuple always
        # axis = 0 if axis == "time" else axis
        if (axis == "spatial") or (axis == "space"):
            if len(data_shape) == 1:
                if dims[0][0] == "t":
                    raise ValueError(f"space axis cannot be selected from dims {dims}")
                return 0
            if "frequency" in dims or "directions" in dims:
                space_name = "node" if "node" in dims else "element"
                return dims.index(space_name)
            else:
                axis = 1 if (len(data_shape) == 2) else tuple(range(1, len(data_shape)))
        if axis is None:
            axis = 0 if (len(data_shape) == 1) else tuple(range(0, len(data_shape)))

        if isinstance(axis, str):
            axis = "time" if axis == "t" else axis
            if axis in dims:
                return dims.index(axis)
            else:
                raise ValueError(
                    f"axis argument '{axis}' not supported! Must be None, int, list of int or 'time' or 'space'"
                )

        return axis

    @staticmethod
    def _axis_to_spatial_axis(dims, axis):
        # subtract 1 if has time axis; assumes axis is integer
        return axis - int(dims[0] == "time")

    @staticmethod
    def _parse_interp_time(old_time, new_time):
        if isinstance(new_time, pd.DatetimeIndex):
            t_out_index = new_time
        elif hasattr(new_time, "time"):
            t_out_index = pd.DatetimeIndex(new_time.time)
        else:
            # offset = pd.tseries.offsets.DateOffset(seconds=new_time) # This seems identical, but doesn't work with slicing
            offset = pd.Timedelta(seconds=new_time)
            t_out_index = pd.date_range(
                start=old_time[0], end=old_time[-1], freq=offset
            )

        return t_out_index

    @staticmethod
    def _interpolate_time(
        intime,
        outtime,
        data: np.ndarray,
        method: str | int,
        extrapolate: bool,
        fill_value: float,
    ):
        from scipy.interpolate import interp1d  # type: ignore

        interpolator = interp1d(
            intime,
            data,
            axis=0,
            kind=method,
            bounds_error=not extrapolate,
            fill_value=fill_value,
        )
        return interpolator(outtime)
