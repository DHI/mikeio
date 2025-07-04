from __future__ import annotations
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Callable

import numpy as np
import pandas as pd

from .dataset import Dataset
from .dfs import Dfs0
from .eum import ItemInfo, EUMUnit, EUMType
from .spatial import GeometryFM2D


def _extract_track(
    *,
    deletevalue: float,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    timestep: float,
    geometry: GeometryFM2D,
    track: str | Path | Dataset | pd.DataFrame,
    items: Sequence[ItemInfo],
    item_numbers: Sequence[int],
    time_steps: Sequence[int],
    n_elements: int,
    method: str,
    dtype: Any,  # TODO DTypeLike?
    data_read_func: Callable[[int, int], tuple[np.ndarray, float]],
) -> Dataset:
    if not isinstance(geometry, GeometryFM2D):
        raise NotImplementedError("Only implemented for 2d flexible mesh geometries")

    n_items = len(item_numbers)

    match track:
        case str():
            times, coords = _get_track_data_from_file(track)
        case Path():
            times, coords = _get_track_data_from_file(str(track))
        case Dataset():
            times, coords = _get_track_data_from_dataset(track)
        case pd.DataFrame():
            times, coords = _get_track_data_from_dataframe(track)
        case _:
            raise ValueError(
                "track must be a file name, a Dataset or a pandas DataFrame"
            )

    assert isinstance(
        times, pd.DatetimeIndex
    ), "The index must be a pandas.DatetimeIndex"
    assert times.is_monotonic_increasing, "The time index must be monotonic increasing. Consider df.sort_index() before passing to extract_track()."

    data_list = [coords[:, 0], coords[:, 1]]  # lon,lat
    for item in range(n_items):
        # Initialize an empty data block
        data = np.empty(shape=(len(times)), dtype=dtype)
        data[:] = np.nan
        data_list.append(data)

    if geometry.is_geo:
        lon = coords[:, 0]
        lon[lon < -180] = lon[lon < -180] + 360
        lon[lon >= 180] = lon[lon >= 180] - 360
        coords[:, 0] = lon

    # track end (relative time)
    t_rel = (times - end_time).total_seconds()
    # largest idx for which (times - self.end_time)<=0
    tmp = np.where(t_rel <= 0)[0]
    if len(tmp) == 0:
        raise ValueError("No time overlap!")
    i_end = tmp[-1]

    # track time relative to start
    t_rel = (times - start_time).total_seconds()
    tmp = np.where(t_rel >= 0)[0]
    if len(tmp) == 0:
        raise ValueError("No time overlap!")
    i_start = tmp[0]  # smallest idx for which t_rel>=0

    dfsu_step = int(np.floor(t_rel[i_start] / timestep))  # first step

    # spatial interpolation
    n_pts = 1 if method == "nearest" else 5
    interpolant = geometry.get_2d_interpolant(
        coords[i_start : (i_end + 1)], n_nearest=n_pts
    )

    # initialize arrays
    d1: np.ndarray = np.ndarray(shape=(n_items, n_elements), dtype=dtype)
    d2: np.ndarray = np.ndarray(shape=(n_items, n_elements), dtype=dtype)
    t1 = 0.0
    t2 = 0.0

    # very first dfsu time step
    step = time_steps[dfsu_step]
    for i, item in enumerate(item_numbers):
        d, t2 = data_read_func(item, step)
        t2 = t2 - 1e-10  # TODO what is this operation doing?
        d[d == deletevalue] = np.nan
        d2[i, :] = d

    def is_EOF(step: int) -> bool:
        return step >= len(time_steps)

    # loop over track points
    for i_interp, t in enumerate(range(i_start, i_end + 1)):
        t_rel[t]  # time of point relative to dfsu start

        read_next = t_rel[t] > t2

        while read_next and not is_EOF(dfsu_step + 1):
            dfsu_step = dfsu_step + 1

            # swap new to old
            d1, d2 = d2, d1
            t1, t2 = t2, t1

            step = time_steps[dfsu_step]
            for i, item in enumerate(item_numbers):
                d, t2 = data_read_func(item, step)
                d[d == deletevalue] = np.nan
                d2[i, :] = d

            read_next = t_rel[t] > t2

        if read_next and is_EOF(dfsu_step):
            # cannot read next - no more timesteps
            continue

        w = (t_rel[t] - t1) / timestep  # time-weight
        eid = interpolant.ids[i_interp]
        weights = interpolant.weights
        # TODO move to interpolation module?
        if np.any(eid > 0):
            dati = (1 - w) * np.dot(d1[:, eid], weights[i_interp])
            dati = dati + w * np.dot(d2[:, eid], weights[i_interp])
        else:
            dati = np.empty(shape=n_items, dtype=dtype)
            dati[:] = np.nan

        for item in range(n_items):
            data_list[item + 2][t] = dati[item]

    if geometry.is_geo:
        items_out = [
            ItemInfo("Longitude", EUMType.Latitude_longitude, EUMUnit.degree),
            ItemInfo("Latitude", EUMType.Latitude_longitude, EUMUnit.degree),
        ]
    else:
        items_out = [
            ItemInfo("x", EUMType.Geographical_coordinate, EUMUnit.meter),
            ItemInfo("y", EUMType.Geographical_coordinate, EUMUnit.meter),
        ]

    for item_info in items:
        items_out.append(item_info)

    return Dataset.from_numpy(data=data_list, time=times, items=items_out)


def _get_track_data_from_dataset(track: Dataset) -> tuple[pd.DatetimeIndex, np.ndarray]:
    times = track.time
    coords = np.zeros(shape=(len(times), 2))
    coords[:, 0] = track[0].to_numpy().copy()
    coords[:, 1] = track[1].to_numpy().copy()
    return times, coords


def _get_track_data_from_dataframe(
    track: pd.DataFrame,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    times = track.index
    coords = track.iloc[:, 0:2].to_numpy(copy=True)
    return times, coords


def _get_track_data_from_file(track: str) -> tuple[pd.DatetimeIndex, np.ndarray]:
    filename = track
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"{filename} does not exist")

    ext = path.suffix.lower()
    match ext:
        case ".dfs0":
            df = Dfs0(filename).to_dataframe()
        case ".csv":
            df = pd.read_csv(filename, index_col=0, parse_dates=True)
            df.index = pd.DatetimeIndex(df.index)
        case _:
            raise ValueError(f"{ext} files not supported (dfs0, csv)")

    times = df.index
    coords = df.iloc[:, 0:2].to_numpy(copy=True)

    return times, coords
