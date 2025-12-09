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


def test_scan_dfs_filter_with_datetime_strings(tmp_path: Path) -> None:
    """Test time filtering with datetime strings."""
    infilename = "tests/testdata/da_diagnostic.dfs0"
    outfile = tmp_path / "filtered_datetime.dfs0"

    # Filter using datetime strings
    scan_dfs(infilename).filter(time=slice("2017-10-27", "2017-10-28")).to_dfs(outfile)

    # Verify
    result = mikeio.read(outfile)
    assert result.time[0].strftime("%Y-%m-%d") == "2017-10-27"
    assert result.time[-1].strftime("%Y-%m-%d") <= "2017-10-28"


def test_scan_dfs_filter_with_step(tmp_path: Path) -> None:
    """Test filtering with step parameter to skip timesteps."""
    infilename = "tests/testdata/HD2D.dfsu"
    outfile = tmp_path / "filtered_step.dfsu"

    # Take every 2nd timestep (0, 2, 4, 6, 8) - HD2D has 9 timesteps
    scan_dfs(infilename).select([0]).filter(time=slice(0, 9, 2)).to_dfs(outfile)

    # Verify
    org = mikeio.read(infilename)
    result = mikeio.read(outfile)

    assert len(result.time) == 5  # 0, 2, 4, 6, 8
    np.testing.assert_array_equal(result[0].to_numpy(), org[0].to_numpy()[0:9:2])


def test_scan_dfs_rolling_with_min_periods(tmp_path: Path) -> None:
    """Test rolling with custom min_periods."""
    infilename = "tests/testdata/random.dfs0"
    outfile = tmp_path / "rolling_min_periods.dfs0"

    # Rolling window with smaller min_periods
    scan_dfs(infilename).select([0]).rolling(
        window=5, stat="mean", min_periods=1
    ).to_dfs(outfile)

    # Verify - should output from first timestep
    org = mikeio.read(infilename)
    result = mikeio.read(outfile)

    assert len(result.time) == len(org.time)  # All timesteps included


def test_scan_dfs_rolling_with_center(tmp_path: Path) -> None:
    """Test rolling with centered window."""
    infilename = "tests/testdata/random.dfs0"
    outfile = tmp_path / "rolling_center.dfs0"

    # Centered rolling window
    scan_dfs(infilename).select([0]).rolling(window=3, stat="mean", center=True).to_dfs(
        outfile
    )

    # Verify structure
    result = mikeio.read(outfile)
    assert len(result.items) == 1


def test_scan_dfs_select_multiple_items(tmp_path: Path) -> None:
    """Test selecting multiple items."""
    infilename = "tests/testdata/random.dfs0"
    outfile = tmp_path / "selected_multiple.dfs0"

    # Select first two items
    scan_dfs(infilename).select([0, 1]).to_dfs(outfile)

    # Verify
    org = mikeio.read(infilename)
    result = mikeio.read(outfile)

    assert len(result.items) == 2
    np.testing.assert_array_equal(result[0].to_numpy(), org[0].to_numpy())
    np.testing.assert_array_equal(result[1].to_numpy(), org[1].to_numpy())


def test_scan_dfs_works_with_dfsu(tmp_path: Path) -> None:
    """Test lazy API works with dfsu files."""
    infilename = "tests/testdata/wind_north_sea.dfsu"
    outfile = tmp_path / "wind_subset.dfsu"

    # Select and filter dfsu
    scan_dfs(infilename).select([0]).filter(time=slice(0, 3)).to_dfs(outfile)

    # Verify
    result = mikeio.read(outfile)
    assert len(result.items) == 1
    assert len(result.time) == 3


def test_scan_dfs_works_with_dfs2(tmp_path: Path) -> None:
    """Test lazy API works with dfs2 files."""
    infilename = "tests/testdata/eq.dfs2"
    outfile = tmp_path / "eq_subset.dfs2"

    # Select and filter dfs2
    scan_dfs(infilename).select([0]).filter(time=slice(0, 2)).to_dfs(outfile)

    # Verify
    result = mikeio.read(outfile)
    assert len(result.items) == 1
    assert len(result.time) == 2


def test_scan_dfs_scale(tmp_path: Path) -> None:
    """Test scale operation with factor and offset."""
    infilename = "tests/testdata/HD2D.dfsu"
    outfile = tmp_path / "scaled.dfsu"

    # Scale: multiply by 2 and add 10
    scan_dfs(infilename).select([0]).scale(factor=2.0, offset=10.0).to_dfs(outfile)

    # Verify
    org = mikeio.read(infilename)
    result = mikeio.read(outfile)

    expected = org[0].to_numpy() * 2.0 + 10.0
    np.testing.assert_array_almost_equal(result[0].to_numpy(), expected)


def test_scan_dfs_aggregate_mean(tmp_path: Path) -> None:
    """Test aggregate operation with mean statistic."""
    infilename = "tests/testdata/HD2D.dfsu"
    outfile = tmp_path / "mean.dfsu"

    # Compute temporal mean
    scan_dfs(infilename).select([0]).aggregate(stat="mean").to_dfs(outfile)

    # Verify
    org = mikeio.read(infilename)
    result = mikeio.read(outfile)

    # Should have single timestep with mean values
    assert len(result.time) == 1
    expected = np.nanmean(org[0].to_numpy(), axis=0)
    np.testing.assert_array_almost_equal(result[0].to_numpy()[0], expected)


def test_scan_dfs_aggregate_min_max(tmp_path: Path) -> None:
    """Test aggregate operation with min and max statistics."""
    infilename = "tests/testdata/HD2D.dfsu"
    outfile_min = tmp_path / "min.dfsu"
    outfile_max = tmp_path / "max.dfsu"

    # Compute temporal min and max
    scan_dfs(infilename).select([0]).aggregate(stat="min").to_dfs(outfile_min)
    scan_dfs(infilename).select([0]).aggregate(stat="max").to_dfs(outfile_max)

    # Verify
    org = mikeio.read(infilename)
    result_min = mikeio.read(outfile_min)
    result_max = mikeio.read(outfile_max)

    expected_min = np.nanmin(org[0].to_numpy(), axis=0)
    expected_max = np.nanmax(org[0].to_numpy(), axis=0)

    np.testing.assert_array_almost_equal(result_min[0].to_numpy()[0], expected_min)
    np.testing.assert_array_almost_equal(result_max[0].to_numpy()[0], expected_max)


def test_scan_dfs_aggregate_dfsu_3d(tmp_path: Path) -> None:
    """Test aggregate operation with dfsu 3d file (layered mesh)."""
    infilename = "tests/testdata/oresund_sigma_z.dfsu"
    outfile = tmp_path / "mean_3d.dfsu"

    # Compute temporal mean - select first non-Z item (Temperature)
    scan_dfs(infilename).select([0]).aggregate(stat="mean").to_dfs(outfile)

    # Verify
    org = mikeio.read(infilename)
    result = mikeio.read(outfile)

    # Should have single timestep
    assert len(result.time) == 1
    # Should have only the selected item (Z coordinate is hidden by mikeio.read)
    assert len(result.items) == 1
    assert result.items[0].name == "Temperature"

    # Verify aggregation result
    expected = np.nanmean(org["Temperature"].to_numpy(), axis=0)
    np.testing.assert_array_almost_equal(result["Temperature"].to_numpy()[0], expected)


def test_lazy_repr() -> None:
    """Test that LazyDfs has a useful repr."""
    infilename = "tests/testdata/HD2D.dfsu"

    # Simple pipeline
    lazy = scan_dfs(infilename).select([0])
    repr_str = repr(lazy)
    assert "LazyDfs" in repr_str
    assert "HD2D.dfsu" in repr_str
    assert "select" in repr_str.lower()

    # Complex pipeline
    lazy = (
        scan_dfs(infilename)
        .select([0, 1])
        .filter(time=slice(0, 5))
        .scale(factor=2.0)
        .aggregate(stat="mean")
    )
    repr_str = repr(lazy)
    assert "select" in repr_str.lower()
    assert "filter" in repr_str.lower()
    assert "scale" in repr_str.lower()
    assert "aggregate" in repr_str.lower()


def test_lazy_explain() -> None:
    """Test that LazyDfs.explain() provides detailed pipeline information."""
    infilename = "tests/testdata/HD2D.dfsu"

    lazy = (
        scan_dfs(infilename)
        .select([0])
        .filter(time=slice("1985-08-06", "1985-08-07"))
        .scale(factor=2.0, offset=10.0)
    )

    explanation = lazy.explain()

    # Should contain operation details
    assert "select" in explanation.lower()
    assert "filter" in explanation.lower()
    assert "scale" in explanation.lower()
    assert "2.0" in explanation  # Factor value
    assert "10.0" in explanation  # Offset value
    assert "1985-08-06" in explanation  # Start time


def test_scan_dfs_diff(tmp_path: Path) -> None:
    """Test diff operation to compute difference between two files."""
    infilename = "tests/testdata/HD2D.dfsu"
    outfile = tmp_path / "diff.dfsu"

    # Compute difference: file - itself should give zeros
    scan_dfs(infilename).select([0]).diff(infilename).to_dfs(outfile)

    # Verify
    result = mikeio.read(outfile)

    # Difference of file with itself should be all zeros (or very close)
    assert np.allclose(result[0].to_numpy(), 0.0, atol=1e-5)


def test_scan_dfs_diff_with_scale(tmp_path: Path) -> None:
    """Test diff operation combined with scale."""
    infilename = "tests/testdata/HD2D.dfsu"
    scaled_file = tmp_path / "scaled.dfsu"
    diff_file = tmp_path / "diff.dfsu"

    # Create a scaled version
    scan_dfs(infilename).select([0]).scale(factor=2.0).to_dfs(scaled_file)

    # Compute difference: scaled - original
    scan_dfs(scaled_file).diff(infilename).to_dfs(diff_file)

    # Verify: difference should be equal to the original data
    org = mikeio.read(infilename)
    result = mikeio.read(diff_file)

    expected = org[0].to_numpy()  # scaled - original = 2*original - original = original
    np.testing.assert_array_almost_equal(result[0].to_numpy(), expected)
