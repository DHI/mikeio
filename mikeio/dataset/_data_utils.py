from __future__ import annotations
from datetime import datetime
import re
from collections.abc import Iterable, Sized

import pandas as pd

from .._time import DateTimeSelector


def _to_safe_name(name: str) -> str:
    tmp = re.sub("[^0-9a-zA-Z]", "_", name)
    return re.sub("_+", "_", tmp)  # Collapse multiple underscores


def _n_selected_timesteps(x: Sized, k: slice | Sized) -> int:
    if isinstance(k, slice):
        k = list(range(*k.indices(len(x))))
    return len(k)


def _get_time_idx_list(
    time: pd.DatetimeIndex,
    steps: int | Iterable[int] | str | datetime | pd.DatetimeIndex | slice,
) -> list[int] | slice:
    """Find list of idx in DatetimeIndex."""
    # indexing with a slice needs to be handled differently, since slicing returns a view

    if isinstance(steps, slice):
        if isinstance(steps.start, int) and isinstance(steps.stop, int):
            return steps

    dts = DateTimeSelector(time)
    return dts.isel(steps)
