from typing import Iterable, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime

# from copy import deepcopy
# from mikeio.eum import ItemInfo


def _time_by_agg_axis(
    time: pd.DatetimeIndex, axis: Union[int, Sequence[int]]
) -> pd.DatetimeIndex:
    """New DatetimeIndex after aggregating over axis"""
    if axis == 0:
        time = pd.DatetimeIndex([time[0]])
    elif isinstance(axis, Sequence) and 0 in axis:
        time = pd.DatetimeIndex([time[0]])
    else:
        time = time

    return time


def _get_time_idx_list(time: pd.DatetimeIndex, steps):
    """Find list of idx in DatetimeIndex"""
    # TODO: allow steps to be other DateTimeAxis
    if isinstance(steps, pd.DatetimeIndex):
        return time.get_indexer(steps)
    if isinstance(steps, (str, datetime, np.datetime64, pd.Timestamp)):
        steps = slice(steps, steps)
    if isinstance(steps, slice):
        try:
            s = time.slice_indexer(steps.start, steps.stop)
            steps = list(range(s.start, s.stop))
        except:
            steps = list(range(*steps.indices(len(time))))
    elif isinstance(steps, int):
        steps = [steps]

    return steps


def _is_boolean_mask(x):
    if hasattr(x, "dtype"):  # isinstance(x, (np.ndarray, DataArray)):
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


def _parse_time(time, data_shape=None):
    """Allow anything that we can create a DatetimeIndex from"""
    if time is None:
        time = [pd.Timestamp(2018, 1, 1)]
    if isinstance(time, str) or (not isinstance(time, Iterable)):
        time = [time]

    if not isinstance(time, pd.DatetimeIndex):
        time = pd.DatetimeIndex(time)

    if data_shape is not None:
        if (len(time) > 1) and data_shape[0] != len(time):
            raise ValueError(
                f"Number of timesteps ({len(time)}) does not fit with data shape {data_shape}"
            )

    if not time.is_monotonic_increasing:
        raise ValueError(
            "Time must be monotonic increasing (only equal or increasing) instances."
        )
    return time


def _parse_axis(data_shape, dims, axis) -> Union[int, Tuple[int]]:
    # axis = 0 if axis == "time" else axis
    if (axis == "spatial") or (axis == "space"):
        if len(data_shape) == 1:
            raise ValueError(
                f"axis '{axis}' not allowed for Dataset with shape {data_shape}"
            )
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


def _axis_to_spatial_axis(dims, axis):
    # subtract 1 if has time axis; assumes axis is integer
    return axis - int(dims[0] == "time")


def _to_safe_name(name):
    return "".join([x if x.isalnum() else "_" for x in name])


def _keepdims_by_axis(axis):
    # keepdims: input to numpy aggregate function
    if axis == 0:
        keepdims = True
    else:
        keepdims = False
    return keepdims


def _reshape_data_by_axis(data, orig_shape, axis):
    if isinstance(axis, int):
        return data
    if len(orig_shape) == len(axis):
        shape = (1,)
        data = [d.reshape(shape) for d in data]
    if len(orig_shape) - len(axis) == 1:
        # e.g. (0,2) for for dfs2
        shape = [1] if (0 in axis) else [orig_shape[0]]
        ndims = len(orig_shape)
        for j in range(1, ndims):
            if j not in axis:
                shape.append(orig_shape[j])
        data = [d.reshape(shape) for d in data]

    return data
