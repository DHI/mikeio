from pathlib import Path

import numpy as np

import mikeio
from mikeio.generic import DerivedItem
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
    """Test rolling mean operation on dfsu file."""
    infilename = "tests/testdata/HD2D.dfsu"
    outfile = tmp_path / "rolling.dfsu"

    # Apply 3-timestep rolling mean
    scan_dfs(infilename).select([0]).rolling(window=3, stat="mean").to_dfs(outfile)

    # Verify structure
    org = mikeio.read(infilename)
    result = mikeio.read(outfile)

    assert len(result.items) == 1
    # Rolling with window=3 and min_periods=3 (default) starts outputting after 2 timesteps
    assert len(result.time) == len(org.time) - 2
    # Verify spatial dimensions preserved
    assert result.geometry.n_elements == org.geometry.n_elements


def test_scan_dfs_rolling_custom_function(tmp_path: Path) -> None:
    """Test rolling with custom function on dfsu file."""
    infilename = "tests/testdata/HD2D.dfsu"
    outfile = tmp_path / "rolling_custom.dfsu"

    # Apply rolling maximum
    scan_dfs(infilename).select([0]).rolling(window=3, stat=np.nanmax).to_dfs(outfile)

    # Verify it runs without error
    result = mikeio.read(outfile)
    assert len(result.items) == 1
    org = mikeio.read(infilename)
    assert result.geometry.n_elements == org.geometry.n_elements


def test_scan_dfs_complete_pipeline(tmp_path: Path) -> None:
    """Test a complete pipeline with multiple operations on dfsu file."""
    infilename = "tests/testdata/HD2D.dfsu"
    outfile = tmp_path / "pipeline.dfsu"

    org = mikeio.read(infilename)
    item_name = org.items[0].name

    # Complete pipeline: select -> filter -> rolling -> transform
    (
        scan_dfs(infilename)
        .select([0])
        .filter(time=slice(0, 7))  # HD2D has 9 timesteps
        .rolling(window=3, stat="mean")
        .with_items(**{item_name: lambda x: x + 100.0})
        .to_dfs(outfile)
    )

    # Verify it completes successfully
    result = mikeio.read(outfile)
    assert len(result.items) == 1
    # Filter 0:7 gives 7 timesteps, rolling with window=3 removes first 2, so 7 -> 5
    assert len(result.time) == 5
    assert result.geometry.n_elements == org.geometry.n_elements


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
    """Test rolling with custom min_periods on dfsu file."""
    infilename = "tests/testdata/HD2D.dfsu"
    outfile = tmp_path / "rolling_min_periods.dfsu"

    # Rolling window with smaller min_periods
    scan_dfs(infilename).select([0]).rolling(
        window=5, stat="mean", min_periods=1
    ).to_dfs(outfile)

    # Verify - should output from first timestep
    org = mikeio.read(infilename)
    result = mikeio.read(outfile)

    assert len(result.time) == len(org.time)  # All timesteps included
    assert result.geometry.n_elements == org.geometry.n_elements


def test_scan_dfs_rolling_with_center(tmp_path: Path) -> None:
    """Test rolling with centered window on dfsu file."""
    infilename = "tests/testdata/HD2D.dfsu"
    outfile = tmp_path / "rolling_center.dfsu"

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
    scan_dfs(infilename).select([0]).aggregate("mean").to_dfs(outfile)

    # Verify
    org = mikeio.read(infilename)
    result = mikeio.read(outfile)

    # Should have single timestep with mean values
    assert len(result.time) == 1
    assert len(result.items) == 1
    assert result.items[0].name == f"Mean: {org.items[0].name}"
    expected = np.nanmean(org[0].to_numpy(), axis=0)
    np.testing.assert_array_almost_equal(result[0].to_numpy()[0], expected)


def test_scan_dfs_aggregate_chunked_processing(tmp_path: Path) -> None:
    """Test aggregate with small buffer_size to verify chunked processing works."""
    infilename = "tests/testdata/wind_north_sea.dfsu"

    # Test with various buffer sizes - all should produce identical results
    org = mikeio.read(infilename)
    expected = np.nanmean(org[0].to_numpy(), axis=0)

    for buffer_size in [100, 1000, 10000, int(1e9)]:
        outfile = tmp_path / f"mean_{buffer_size}.dfsu"
        scan_dfs(infilename).select([0]).aggregate("mean").to_dfs(
            outfile, buffer_size=buffer_size
        )

        result = mikeio.read(outfile)
        assert len(result.time) == 1
        assert result.items[0].name == f"Mean: {org.items[0].name}"
        np.testing.assert_array_almost_equal(result[0].to_numpy()[0], expected)


def test_scan_dfs_aggregate_min_max(tmp_path: Path) -> None:
    """Test aggregate operation with multiple statistics in single pass."""
    infilename = "tests/testdata/HD2D.dfsu"
    outfile = tmp_path / "stats.dfsu"

    # Compute multiple statistics in one pass
    scan_dfs(infilename).select([0]).aggregate(["min", "max", "mean", "std"]).to_dfs(
        outfile
    )

    # Verify
    org = mikeio.read(infilename)
    result = mikeio.read(outfile)

    # Should have single timestep with 4 items (one for each stat)
    assert len(result.time) == 1
    assert len(result.items) == 4

    # Check item names follow "Stat: ItemName" convention
    orig_name = org.items[0].name
    expected_names = [
        f"Min.: {orig_name}",
        f"Max.: {orig_name}",
        f"Mean: {orig_name}",
        f"Std.: {orig_name}",
    ]
    result_names = [item.name for item in result.items]
    assert result_names == expected_names

    # Verify values for each statistic
    expected_min = np.nanmin(org[0].to_numpy(), axis=0)
    expected_max = np.nanmax(org[0].to_numpy(), axis=0)
    expected_mean = np.nanmean(org[0].to_numpy(), axis=0)
    expected_std = np.nanstd(org[0].to_numpy(), axis=0)

    np.testing.assert_array_almost_equal(result[0].to_numpy()[0], expected_min)
    np.testing.assert_array_almost_equal(result[1].to_numpy()[0], expected_max)
    np.testing.assert_array_almost_equal(result[2].to_numpy()[0], expected_mean)
    np.testing.assert_array_almost_equal(result[3].to_numpy()[0], expected_std)


def test_scan_dfs_aggregate_dfsu_3d(tmp_path: Path) -> None:
    """Test aggregate operation with dfsu 3d file (layered mesh)."""
    infilename = "tests/testdata/oresund_sigma_z.dfsu"
    outfile = tmp_path / "mean_3d.dfsu"

    # Compute temporal mean - select first non-Z item (Temperature)
    scan_dfs(infilename).select([0]).aggregate("mean").to_dfs(outfile)

    # Verify
    org = mikeio.read(infilename)
    result = mikeio.read(outfile)

    # Should have single timestep
    assert len(result.time) == 1
    # Should have only the selected item (Z coordinate is hidden by mikeio.read)
    assert len(result.items) == 1
    assert result.items[0].name == "Mean: Temperature"

    # Verify aggregation result
    expected = np.nanmean(org["Temperature"].to_numpy(), axis=0)
    np.testing.assert_array_almost_equal(result[0].to_numpy()[0], expected)


def test_scan_dfs_aggregate_multiple_items_multiple_stats(tmp_path: Path) -> None:
    """Test aggregate with multiple items and multiple statistics."""
    infilename = "tests/testdata/wind_north_sea.dfsu"
    outfile = tmp_path / "multi_stats.dfsu"

    # Compute multiple stats for 2 items
    scan_dfs(infilename).select([0, 1]).aggregate(["min", "max"]).to_dfs(outfile)

    # Verify
    org = mikeio.read(infilename)
    result = mikeio.read(outfile)

    # Should have single timestep with 4 items (2 items × 2 stats)
    assert len(result.time) == 1
    assert len(result.items) == 4

    # Check item names: Min./Max. for each item
    expected_names = [
        f"Min.: {org.items[0].name}",
        f"Max.: {org.items[0].name}",
        f"Min.: {org.items[1].name}",
        f"Max.: {org.items[1].name}",
    ]
    result_names = [item.name for item in result.items]
    assert result_names == expected_names

    # Verify values
    expected_min_0 = np.nanmin(org[0].to_numpy(), axis=0)
    expected_max_0 = np.nanmax(org[0].to_numpy(), axis=0)
    expected_min_1 = np.nanmin(org[1].to_numpy(), axis=0)
    expected_max_1 = np.nanmax(org[1].to_numpy(), axis=0)

    np.testing.assert_array_almost_equal(result[0].to_numpy()[0], expected_min_0)
    np.testing.assert_array_almost_equal(result[1].to_numpy()[0], expected_max_0)
    np.testing.assert_array_almost_equal(result[2].to_numpy()[0], expected_min_1)
    np.testing.assert_array_almost_equal(result[3].to_numpy()[0], expected_max_1)


def test_scan_dfs_aggregate_custom_labels(tmp_path: Path) -> None:
    """Test aggregate with custom stat labels."""
    infilename = "tests/testdata/HD2D.dfsu"
    outfile = tmp_path / "custom_labels.dfsu"

    # Use custom labels for stats
    scan_dfs(infilename).select([0]).aggregate(
        ["min", "max", "mean"],
        labels={"mean": "Avg", "max": "Maximum", "min": "Minimum"},
    ).to_dfs(outfile)

    # Verify
    org = mikeio.read(infilename)
    result = mikeio.read(outfile)

    # Check custom labels are used
    orig_name = org.items[0].name
    expected_names = [
        f"Minimum: {orig_name}",
        f"Maximum: {orig_name}",
        f"Avg: {orig_name}",
    ]
    result_names = [item.name for item in result.items]
    assert result_names == expected_names

    # Verify values are still correct
    expected_min = np.nanmin(org[0].to_numpy(), axis=0)
    expected_max = np.nanmax(org[0].to_numpy(), axis=0)
    expected_mean = np.nanmean(org[0].to_numpy(), axis=0)

    np.testing.assert_array_almost_equal(result[0].to_numpy()[0], expected_min)
    np.testing.assert_array_almost_equal(result[1].to_numpy()[0], expected_max)
    np.testing.assert_array_almost_equal(result[2].to_numpy()[0], expected_mean)


def test_scan_dfs_aggregate_custom_format(tmp_path: Path) -> None:
    """Test aggregate with custom format string."""
    infilename = "tests/testdata/HD2D.dfsu"
    outfile = tmp_path / "custom_format.dfsu"

    # Use suffix style format
    scan_dfs(infilename).select([0]).aggregate(
        ["min", "max"], format="{item} {stat}"
    ).to_dfs(outfile)

    # Verify
    org = mikeio.read(infilename)
    result = mikeio.read(outfile)

    # Check suffix format is used
    orig_name = org.items[0].name
    expected_names = [
        f"{orig_name} Min.",
        f"{orig_name} Max.",
    ]
    result_names = [item.name for item in result.items]
    assert result_names == expected_names

    # Verify values
    expected_min = np.nanmin(org[0].to_numpy(), axis=0)
    expected_max = np.nanmax(org[0].to_numpy(), axis=0)

    np.testing.assert_array_almost_equal(result[0].to_numpy()[0], expected_min)
    np.testing.assert_array_almost_equal(result[1].to_numpy()[0], expected_max)


def test_scan_dfs_aggregate_custom_labels_and_format(tmp_path: Path) -> None:
    """Test aggregate with both custom labels and format."""
    infilename = "tests/testdata/HD2D.dfsu"
    outfile = tmp_path / "custom_both.dfsu"

    # Use custom labels AND format
    scan_dfs(infilename).select([0]).aggregate(
        ["min", "max", "mean"],
        labels={"mean": "Avg", "max": "Max", "min": "Min"},
        format="{item} ({stat})",
    ).to_dfs(outfile)

    # Verify
    org = mikeio.read(infilename)
    result = mikeio.read(outfile)

    # Check both custom labels and format are used
    orig_name = org.items[0].name
    expected_names = [
        f"{orig_name} (Min)",
        f"{orig_name} (Max)",
        f"{orig_name} (Avg)",
    ]
    result_names = [item.name for item in result.items]
    assert result_names == expected_names

    # Verify values
    expected_min = np.nanmin(org[0].to_numpy(), axis=0)
    expected_max = np.nanmax(org[0].to_numpy(), axis=0)
    expected_mean = np.nanmean(org[0].to_numpy(), axis=0)

    np.testing.assert_array_almost_equal(result[0].to_numpy()[0], expected_min)
    np.testing.assert_array_almost_equal(result[1].to_numpy()[0], expected_max)
    np.testing.assert_array_almost_equal(result[2].to_numpy()[0], expected_mean)


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
        .filter(time=slice(0, 5))  # Use step indices instead of datetime
        .scale(factor=2.0, offset=10.0)
    )

    explanation = lazy.explain()

    # Should contain operation details
    assert "select" in explanation.lower()
    assert "filter" in explanation.lower()
    assert "scale" in explanation.lower()
    assert "2.0" in explanation  # Factor value
    assert "10.0" in explanation  # Offset value
    assert "1985-08-06" in explanation  # File start time
    assert "validated successfully" in explanation.lower()  # Validation confirmation


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


def test_validation_invalid_stat_name() -> None:
    """Test that invalid stat names are caught immediately."""
    import pytest

    with pytest.raises(ValueError, match="Unknown stat 'invalid_stat'"):
        scan_dfs("tests/testdata/HD2D.dfsu").aggregate("invalid_stat")

    with pytest.raises(ValueError, match="Unknown stat 'bad_rolling'"):
        scan_dfs("tests/testdata/HD2D.dfsu").rolling(window=3, stat="bad_rolling")


def test_validation_invalid_item_selection() -> None:
    """Test that invalid item selection is caught in explain()."""
    import pytest

    lazy = scan_dfs("tests/testdata/HD2D.dfsu").select(["NonexistentItem"])

    with pytest.raises(ValueError, match="Invalid item selection"):
        lazy.explain()


def test_validation_invalid_time_range() -> None:
    """Test that invalid time ranges are caught in explain()."""
    import pytest

    # End before start
    lazy = scan_dfs("tests/testdata/HD2D.dfsu").filter(start=10, end=5)

    with pytest.raises(ValueError, match="Invalid time filter"):
        lazy.explain()


def test_validation_invalid_diff_file() -> None:
    """Test that missing diff file is caught in explain()."""
    import pytest

    lazy = scan_dfs("tests/testdata/HD2D.dfsu").diff("nonexistent_file.dfsu")

    with pytest.raises(ValueError, match="Diff file not found"):
        lazy.explain()


def test_validation_in_execute(tmp_path: Path) -> None:
    """Test that validation also happens during to_dfs() execution."""
    import pytest

    outfile = tmp_path / "output.dfsu"
    lazy = scan_dfs("tests/testdata/HD2D.dfsu").select(["InvalidItem"])

    with pytest.raises(ValueError, match="Invalid item selection"):
        lazy.to_dfs(outfile)


def test_scan_dfs_derive_from_multiple_items(tmp_path: Path) -> None:
    """Test deriving new item from multiple existing items."""
    infilename = "tests/testdata/HD2D.dfsu"
    outfile = tmp_path / "derived.dfsu"

    # Derive current speed from U and V velocity
    (
        scan_dfs(infilename)
        .select(["U velocity", "V velocity"])
        .derive(
            DerivedItem(
                name="Derived Speed",
                type=mikeio.EUMType.Current_Speed,
                func=lambda x: np.sqrt(x["U velocity"] ** 2 + x["V velocity"] ** 2),
            )
        )
        .to_dfs(outfile)
    )

    # Verify
    org = mikeio.read(infilename)
    result = mikeio.read(outfile)

    # Should have 3 items: U velocity, V velocity, and Derived Speed
    assert len(result.items) == 3
    assert result.items[0].name == "U velocity"
    assert result.items[1].name == "V velocity"
    assert result.items[2].name == "Derived Speed"
    assert result.items[2].type == mikeio.EUMType.Current_Speed

    # Verify the derived speed matches manual calculation
    u = org["U velocity"].to_numpy()
    v = org["V velocity"].to_numpy()
    expected_speed = np.sqrt(u**2 + v**2)
    np.testing.assert_array_almost_equal(
        result["Derived Speed"].to_numpy(), expected_speed, decimal=5
    )


def test_scan_dfs_derive_with_aggregate(tmp_path: Path) -> None:
    """Test deriving items from aggregated data (composability)."""
    infilename = "tests/testdata/HD2D.dfsu"
    outfile = tmp_path / "aggregated_derived.dfsu"

    # Aggregate U and V to mean, then derive speed from mean velocities
    (
        scan_dfs(infilename)
        .select(["U velocity", "V velocity"])
        .aggregate("mean")
        .derive(
            DerivedItem(
                name="Mean Current Speed",
                type=mikeio.EUMType.Current_Speed,
                func=lambda x: np.sqrt(x["U velocity"] ** 2 + x["V velocity"] ** 2),
            )
        )
        .to_dfs(outfile)
    )

    # Verify
    org = mikeio.read(infilename)
    result = mikeio.read(outfile)

    # Should have single timestep (aggregated)
    assert len(result.time) == 1

    # Should have 3 items
    assert len(result.items) == 3
    assert result.items[2].name == "Mean Current Speed"

    # Verify the derived speed matches manual calculation from aggregated data
    u_mean = org["U velocity"].to_numpy().mean(axis=0)
    v_mean = org["V velocity"].to_numpy().mean(axis=0)
    expected_speed = np.sqrt(u_mean**2 + v_mean**2)
    np.testing.assert_array_almost_equal(
        result["Mean Current Speed"].to_numpy()[0], expected_speed, decimal=5
    )
