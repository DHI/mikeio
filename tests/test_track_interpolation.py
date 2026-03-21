"""Tests for track extraction time interpolation."""

import pandas as pd
import pytest

import mikeio


def test_track_at_exact_timestep() -> None:
    """Track point at exact dfsu timestep should return exact values."""
    dfs = mikeio.open("tests/testdata/HD2D.dfsu")
    ds = mikeio.read("tests/testdata/HD2D.dfsu", items=[3])
    da = ds[0]

    # Use element 100 centroid as track coordinate
    ec = dfs.geometry.element_coordinates
    x, y = ec[100, 0], ec[100, 1]

    # Track at exact first timestep
    t0 = dfs.start_time
    track_df = pd.DataFrame({"x": [x], "y": [y]}, index=pd.DatetimeIndex([t0]))
    result = dfs.extract_track(track_df, items=[3])

    expected = da.to_numpy()[0, 100]
    assert result[2].to_numpy()[0] == pytest.approx(expected, abs=1e-5)


def test_track_between_timesteps() -> None:
    """Track point halfway between timesteps should get linear interpolation."""
    dfs = mikeio.open("tests/testdata/HD2D.dfsu")
    ds = mikeio.read("tests/testdata/HD2D.dfsu", items=[3])
    da = ds[0]

    ec = dfs.geometry.element_coordinates
    x, y = ec[100, 0], ec[100, 1]

    # Track halfway between first two timesteps
    t0 = dfs.start_time
    t_half = t0 + pd.Timedelta(seconds=dfs.timestep / 2)
    track_df = pd.DataFrame({"x": [x], "y": [y]}, index=pd.DatetimeIndex([t_half]))
    result = dfs.extract_track(track_df, items=[3])

    v0 = da.to_numpy()[0, 100]
    v1 = da.to_numpy()[1, 100]
    expected = (v0 + v1) / 2.0

    assert result[2].to_numpy()[0] == pytest.approx(expected, abs=1e-5)
