from __future__ import annotations
from datetime import datetime
from dataclasses import dataclass
from collections.abc import Iterable
from typing import Sized

import pandas as pd


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


@dataclass
class DateTimeSelector:
    """Helper class for selecting time steps from a pandas DatetimeIndex."""

    index: pd.DatetimeIndex

    def isel(
        self,
        key: (
            int | Iterable[int] | str | datetime | pd.DatetimeIndex | slice | None
        ) = None,
        /,
    ) -> list[int]:
        """Select time steps from a pandas DatetimeIndex.

        Parameters
        ----------
        key : int, Iterable[int], str, datetime, pd.DatetimeIndex, slice
            Time steps to select, negative indices are supported

        Returns
        -------
        list[int]
            List of indices in the range (0, len(index)
        Examples
        --------
        ```{python}
        import mikeio
        import pandas as pd
        idx = pd.date_range("2000-01-01", periods=4, freq="D")
        dts = DateTimeSelector(idx)
        dts.isel(None)
        ```

        ```{python}
        dts.isel(0)
        ```

        ```{python}
        dts.isel(-1)
        ```

        """
        indices = list(range(len(self.index)))

        match key:
            case None:
                return indices

            case int():
                return [indices[key]]

            case datetime() | str():
                loc = self.index.get_loc(key)
                match loc:
                    case int():
                        return [loc]
                    case slice():
                        return list(range(loc.start, loc.stop))

            case slice():
                if isinstance(key.start, int) or isinstance(key.stop, int):
                    return indices[key]
                else:
                    s = self.index.slice_indexer(key.start, key.stop)
                    return list(range(s.start, s.stop))

            case Iterable():
                return [self.isel(t)[0] for t in key]

            case _:
                return indices

        return indices
