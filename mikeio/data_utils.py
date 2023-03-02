import re
from datetime import datetime
from typing import Iterable, List, Sequence, Sized, Tuple, Union

import numpy as np
import pandas as pd

from .base import TimeSeries


def _to_safe_name(name: str) -> str:
    tmp = re.sub("[^0-9a-zA-Z]", "_", name)
    return re.sub("_+", "_", tmp)  # Collapse multiple underscores


def _n_selected_timesteps(x: Sized, k: Union[slice, Sized]) -> int:
    if isinstance(k, slice):
        k = list(range(*k.indices(len(x))))
    return len(k)


def _get_time_idx_list(time: pd.DatetimeIndex, steps):
    """Find list of idx in DatetimeIndex"""

    if isinstance(steps, str):
        parts = steps.split(",")
        if len(parts) == 1:
            parts.append(parts[0])  # end=start

        if parts[0] == "":
            steps = slice(parts[1])  # stop only
        elif parts[1] == "":
            steps = slice(parts[0], None)  # start only
        else:
            steps = slice(parts[0], parts[1])

    if (isinstance(steps, (List, Tuple)) and not isinstance(steps, str)) and isinstance(
        steps[0], (str, datetime, np.datetime64, pd.Timestamp)
    ):
        steps = pd.DatetimeIndex(steps)
    if isinstance(steps, pd.DatetimeIndex):
        return time.get_indexer(steps)
    if isinstance(steps, (str, datetime, np.datetime64, pd.Timestamp)):
        steps = slice(steps, steps)
    if isinstance(steps, slice):
        try:
            s = time.slice_indexer(
                steps.start,
                steps.stop,
            )
            steps = list(range(s.start, s.stop))
        except TypeError:
            pass  # TODO this seems fishy!
            # steps = list(range(*steps.indices(len(time))))
    elif isinstance(steps, int):
        steps = [steps]
    # TODO what is the return type of this function
    return steps


class DataUtilsMixin:
    """DataArray Utils"""

    @staticmethod
    def _to_safe_name(name: str) -> str:
        return _to_safe_name(name)

    @staticmethod
    def _time_by_agg_axis(
        time: pd.DatetimeIndex, axis: Union[int, Sequence[int]]
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
            time = [pd.Timestamp(2018, 1, 1)]
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
    def _parse_axis(data_shape, dims, axis) -> Union[int, Tuple[int]]:
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
        elif isinstance(new_time, TimeSeries):
            t_out_index = new_time.time
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
        data: np.array,
        method: Union[str, int],
        extrapolate: bool,
        fill_value: float,
    ):
        from scipy.interpolate import interp1d

        interpolator = interp1d(
            intime,
            data,
            axis=0,
            kind=method,
            bounds_error=not extrapolate,
            fill_value=fill_value,
        )
        return interpolator(outtime)
