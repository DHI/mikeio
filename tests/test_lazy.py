from pathlib import Path

import numpy as np

import mikeio
from mikeio.lazy import scan_dfs


def test_scan_dfs_select(tmp_path: Path) -> None:
    """Test basic select operation."""
    infilename = "tests/testdata/random.dfs0"
    outfile = tmp_path / "selected.dfs0"

    # Select first item only
    scan_dfs(infilename).select([0]).to_dfs(outfile)

    # Verify
    org = mikeio.read(infilename)
    result = mikeio.read(outfile)

    assert len(result.items) == 1
    assert len(org.items) > 1
    assert result.items[0] == org.items[0]
    np.testing.assert_array_equal(result[0].to_numpy(), org[0].to_numpy())


def test_scan_dfs_select_by_name(tmp_path: Path) -> None:
    """Test select operation using item names."""
    infilename = "tests/testdata/wind_north_sea.dfsu"
    outfile = tmp_path / "selected.dfsu"

    # Select by name
    scan_dfs(infilename).select(["Wind speed"]).to_dfs(outfile)

    # Verify
    org = mikeio.read(infilename)
    result = mikeio.read(outfile)

    assert len(result.items) == 1
    assert result.items[0] == org.items[0]
    np.testing.assert_array_equal(
        result["Wind speed"].to_numpy(), org["Wind speed"].to_numpy()
    )


def test_scan_dfs_filter_time(tmp_path: Path) -> None:
    """Test time filtering with slice."""
    infilename = "tests/testdata/random.dfs0"
    outfile = tmp_path / "filtered.dfs0"

    # Filter to first 5 timesteps
    scan_dfs(infilename).filter(time=slice(0, 5)).to_dfs(outfile)

    # Verify
    org = mikeio.read(infilename)
    result = mikeio.read(outfile)

    assert len(result.time) == 5
    assert len(org.time) > 5
    np.testing.assert_array_equal(result[0].to_numpy(), org[0].to_numpy()[:5])


def test_scan_dfs_select_and_filter(tmp_path: Path) -> None:
    """Test combining select and filter operations."""
    infilename = "tests/testdata/random.dfs0"
    outfile = tmp_path / "select_filter.dfs0"

    # Select first item and first 5 timesteps
    (scan_dfs(infilename).select([0]).filter(time=slice(0, 5)).to_dfs(outfile))

    # Verify
    org = mikeio.read(infilename)
    result = mikeio.read(outfile)

    assert len(result.items) == 1
    assert len(result.time) == 5
    np.testing.assert_array_equal(result[0].to_numpy(), org[0].to_numpy()[:5])


def test_scan_dfs_with_items(tmp_path: Path) -> None:
    """Test with_items transformation."""
    infilename = "tests/testdata/random.dfs0"
    outfile = tmp_path / "transformed.dfs0"

    # Scale first item by 2
    org = mikeio.read(infilename)
    item_name = org.items[0].name

    (
        scan_dfs(infilename)
        .select([0])
        .with_items(**{item_name: lambda x: x * 2.0})
        .to_dfs(outfile)
    )

    # Verify
    result = mikeio.read(outfile)

    expected = org[0].to_numpy() * 2.0
    np.testing.assert_array_almost_equal(result[0].to_numpy(), expected)


def test_scan_dfs_rolling_mean(tmp_path: Path) -> None:
    """Test rolling mean operation."""
    infilename = "tests/testdata/random.dfs0"
    outfile = tmp_path / "rolling.dfs0"

    # Apply 3-timestep rolling mean
    scan_dfs(infilename).select([0]).rolling(window=3, stat="mean").to_dfs(outfile)

    # Verify structure
    org = mikeio.read(infilename)
    result = mikeio.read(outfile)

    assert len(result.items) == 1
    # Rolling with window=3 and min_periods=3 (default) starts outputting after 2 timesteps
    assert len(result.time) == len(org.time) - 2


def test_scan_dfs_rolling_custom_function(tmp_path: Path) -> None:
    """Test rolling with custom function."""
    infilename = "tests/testdata/random.dfs0"
    outfile = tmp_path / "rolling_custom.dfs0"

    # Apply rolling maximum
    scan_dfs(infilename).select([0]).rolling(window=3, stat=np.nanmax).to_dfs(outfile)

    # Verify it runs without error
    result = mikeio.read(outfile)
    assert len(result.items) == 1


def test_scan_dfs_complete_pipeline(tmp_path: Path) -> None:
    """Test a complete pipeline with multiple operations."""
    infilename = "tests/testdata/random.dfs0"
    outfile = tmp_path / "pipeline.dfs0"

    org = mikeio.read(infilename)
    item_name = org.items[0].name

    # Complete pipeline: select -> filter -> rolling -> transform
    (
        scan_dfs(infilename)
        .select([0])
        .filter(time=slice(0, 10))
        .rolling(window=3, stat="mean")
        .with_items(**{item_name: lambda x: x + 100.0})
        .to_dfs(outfile)
    )

    # Verify it completes successfully
    result = mikeio.read(outfile)
    assert len(result.items) == 1
    # Rolling with window=3 removes first 2 timesteps, so 10 -> 8
    assert len(result.time) == 8
