from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import mikeio
from mikeio import Dfs0, EUMType, EUMUnit, ItemInfo
from mikecore.DfsFile import DataValueType


import pytest


def test_repr() -> None:
    dfs = Dfs0("tests/testdata/da_diagnostic.dfs0")

    text = repr(dfs)

    assert "NonEquidistant" in text


def test_repr_equidistant() -> None:
    dfs = Dfs0("tests/testdata/random.dfs0")

    text = repr(dfs)

    assert "Dfs0" in text
    assert "Equidistant" in text
    assert "NonEquidistant" not in text


def test_write_float(tmp_path: Path) -> None:
    fp = tmp_path / "simple_float.dfs0"

    nt = 100

    da = mikeio.DataArray(
        data=np.random.random([nt]).astype(np.float32),
        time=pd.date_range("2000", periods=nt, freq="h"),
    )

    da.to_dfs(fp)

    assert fp.exists()


def test_write_double(tmp_path: Path) -> None:
    fp = tmp_path / "simple_float.dfs0"

    nt = 100

    da = mikeio.DataArray(
        data=np.random.random([nt]).astype(np.float32),
        time=pd.date_range("2000", periods=nt, freq="h"),
    )

    da.to_dfs(fp, dtype=np.float64)

    assert fp.exists()


def test_write_int_not_possible(tmp_path: Path) -> None:
    fp = tmp_path / "simple_float.dfs0"

    nt = 100

    da = mikeio.DataArray(
        data=np.random.random([nt]).astype(np.float32),
        time=pd.date_range("2000", periods=nt, freq="h"),
    )

    with pytest.raises(TypeError):
        da.to_dfs(fp, dtype=np.int32)


def test_read_units_write_new(tmp_path: Path) -> None:
    fp = tmp_path / "random.dfs0"

    ds = mikeio.read("tests/testdata/random.dfs0")

    # write new file
    ds.to_dfs(fp)

    # Verify that new file has same variables/units as original
    ds2 = mikeio.read(fp)

    assert ds2.items[0] == ds.items[0]


def test_read_start_end_time() -> None:
    dfs = Dfs0("tests/testdata/random.dfs0")
    ds = dfs.read()

    assert dfs.start_time == ds.start_time
    assert dfs.end_time == ds.end_time


def test_read_all_time_steps_without_reading_data() -> None:
    dfs = mikeio.Dfs0("tests/testdata/random.dfs0")
    assert isinstance(dfs.time, pd.DatetimeIndex)
    assert len(dfs.time) == 1000


def test_items_dataframe() -> None:
    dfs = mikeio.Dfs0("tests/testdata/random.dfs0")
    df = dfs.items.to_dataframe()
    assert "name" in df.columns
    assert "type" in df.columns  # or EUMType ?
    assert df.type.iloc[1] == "Water_Level"  # Is this the correct way to show it?


def test_read_all_time_steps_without_reading_items_neq() -> None:
    dfs = mikeio.Dfs0("tests/testdata/da_diagnostic.dfs0")
    assert isinstance(dfs.time, pd.DatetimeIndex)
    assert len(dfs.time) == 744


def test_write_non_equidistant_calendar(tmp_path: Path) -> None:
    dfs0file = tmp_path / "neq.dfs0"
    time = pd.DatetimeIndex(["2001-01-01", "2001-01-01 01:00", "2001-01-01 01:10"])
    da1 = mikeio.DataArray(
        data=np.zeros(3),
        time=time,
        item=ItemInfo("VarFun01", EUMType.Water_Level, unit=EUMUnit.meter),
    )
    da2 = mikeio.DataArray(
        data=np.ones(3),
        time=time,
        item=ItemInfo("NotFun", EUMType.Rainfall_Depth, data_value_type="Accumulated"),
    )
    ds = mikeio.Dataset([da1, da2])
    ds.to_dfs(dfs0file)
    assert dfs0file.exists()

    ds2 = mikeio.read(dfs0file)
    assert not ds2.is_equidistant


def test_read_equidistant_dfs0_to_dataframe_unit_in_name() -> None:
    dfs = Dfs0("tests/testdata/random.dfs0")
    df = dfs.to_dataframe(unit_in_name=True)

    assert "meter" in df.columns[0]


def test_read_nonequidistant_dfs0_to_dataframe_no_freq() -> None:
    dfs = Dfs0("tests/testdata/da_diagnostic.dfs0")
    df = dfs.to_dataframe()

    assert df.index.freq is None


def test_read_dfs0_delete_value_conversion() -> None:
    dfs = Dfs0("tests/testdata/da_diagnostic.dfs0")
    ds = dfs.read()

    assert np.isnan(ds[3].values[1])

    dfs = Dfs0("tests/testdata/random.dfs0")
    ds = dfs.read()

    assert np.isnan(ds[0].values[2])


def test_read_dfs0_small_value_not_delete_value(tmp_path: Path) -> None:
    filename = tmp_path / "small.dfs0"
    d = np.array([0.0, 0.0000001, -0.0001])
    assert np.isclose(d, -1e-35, atol=1e-33).any()

    da = mikeio.DataArray(
        data=d,
        time=pd.date_range("2000", periods=len(d), freq="h"),
    )

    da.to_dfs(filename)

    ds = mikeio.read(filename)

    assert not np.isnan(ds[0].to_numpy()).any()


def test_write_from_data_frame(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {"Average": [1.0, 2.0, np.nan], "Trend": [0.1, 0.2, 0.3]},
        index=pd.date_range("2000-01-01", periods=3, freq="D"),
    )

    filename = tmp_path / "dataframe.dfs0"
    with pytest.warns(FutureWarning):
        Dfs0.from_dataframe(
            df,
            filename,
            itemtype=EUMType.Concentration,
            unit=EUMUnit.gram_per_meter_pow_3,
        )

    ds = mikeio.read(filename)

    assert len(ds.items) == 2
    assert ds.items[0].type == EUMType.Concentration
    assert ds.items[0].unit == EUMUnit.gram_per_meter_pow_3
    assert ds.items[0].data_value_type == DataValueType.Instantaneous


def test_write_dataframe_different_eum_types_to_dfs0(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {"flow": np.array([1, np.nan, 2]), "level": np.array([2, 3.0, -1.3])},
        index=pd.DatetimeIndex(["2001-01-01", "2001-01-01 01:00", "2001-01-01 01:10"]),
    )

    fp = tmp_path / "dataframe.dfs0"

    mikeio.from_pandas(
        df,
        items=[
            mikeio.ItemInfo(mikeio.EUMType.Discharge),
            mikeio.ItemInfo(mikeio.EUMType.Water_Level),
        ],
    ).to_dfs(fp)

    ds = mikeio.read(fp)
    assert ds["flow"].type == mikeio.EUMType.Discharge
    assert ds["level"].type == mikeio.EUMType.Water_Level
    assert len(ds) == 2
    assert ds.end_time == df.index[-1]


def test_from_pandas_mapping_eum_types() -> None:
    df = pd.DataFrame(
        {"flow": np.array([1, np.nan, 2]), "rain": np.array([2, 3.0, -1.3])},
        index=pd.DatetimeIndex(["2001-01-01", "2001-01-01 01:00", "2001-01-01 01:10"]),
    )

    ds = mikeio.from_pandas(
        df,
        items={
            "flow": mikeio.ItemInfo(itemtype=mikeio.EUMType.Discharge),
            "rain": mikeio.ItemInfo(
                itemtype=mikeio.EUMType.Rainfall,
                unit=mikeio.EUMUnit.centimeter,
                data_value_type="StepAccumulated",
            ),
        },
    )

    assert ds["flow"].type == mikeio.EUMType.Discharge
    assert ds["rain"].type == mikeio.EUMType.Rainfall
    assert ds[0].name == "flow"
    assert ds["rain"].item.data_value_type == DataValueType.StepAccumulated


def test_from_pandas_same_eum_type() -> None:
    df = pd.DataFrame(
        {"station_a": np.array([1, np.nan, 2]), "station_b": np.array([2, 3.0, -1.3])},
        index=pd.date_range("2001-01-01", periods=3, freq="h"),
    )

    ds = mikeio.from_pandas(
        df,
        items=ItemInfo(EUMType.Water_Level, data_value_type=DataValueType.Accumulated),
    )

    assert ds.n_timesteps == 3
    assert ds[0].type == EUMType.Water_Level
    assert ds[0].item.data_value_type == DataValueType.Accumulated
    assert ds["station_b"].item.name == "station_b"


def test_from_pandas_sequence_eum_types() -> None:
    df = pd.DataFrame(
        {"flow": np.array([1, np.nan, 2]), "level": np.array([2, 3.0, -1.3])},
        index=pd.DatetimeIndex(["2001-01-01", "2001-01-01 01:00", "2001-01-01 01:10"]),
    )

    ds = mikeio.from_pandas(
        df,
        items=[
            mikeio.ItemInfo("Ignored", itemtype=mikeio.EUMType.Discharge),
            mikeio.ItemInfo(
                "Also Ignored",
                itemtype=mikeio.EUMType.Water_Level,
                unit=mikeio.EUMUnit.millimeter,
                data_value_type=DataValueType.Accumulated,
            ),
        ],
    )

    assert ds["flow"].type == mikeio.EUMType.Discharge
    assert ds["level"].type == mikeio.EUMType.Water_Level
    assert ds["level"].item.unit == mikeio.EUMUnit.millimeter
    assert ds["level"].item.data_value_type == DataValueType.Accumulated
    assert ds["level"].item.name == "level"


def test_from_pandas_use_first_datetime_column() -> None:
    df = pd.DataFrame(
        {
            "time": pd.date_range("2001-01-01", periods=3, freq="h"),
            "flow": np.array([1, np.nan, 2]),
            "level": np.array([2, 3.0, -1.3]),
        }
    )
    # no index set, uses first datetime column
    ds = mikeio.from_pandas(df)

    assert ds.n_timesteps == 3
    assert ds.time[-1].year == 2001


def test_from_pandas_no_time_raises_error() -> None:
    df = pd.DataFrame(
        {
            "flow": np.array([1, np.nan, 2]),
            "level": np.array([2, 3.0, -1.3]),
        }
    )

    with pytest.raises(ValueError, match="datetime"):
        mikeio.from_pandas(df)


def test_from_polars_explicit_time_column() -> None:
    import polars as pl

    df = pl.DataFrame(
        {
            "flow": [1.0, None, 2.0],
            "level": [2.0, 3.0, -1.3],
            "time": [
                datetime(2001, 1, 1, 0),
                datetime(2001, 1, 1, 1),
                datetime(2001, 1, 1, 2),
            ],
        }
    )

    ds = mikeio.from_polars(
        df,
        datetime_col="time",
        items={
            "flow": ItemInfo(EUMType.Discharge),
            "level": ItemInfo(EUMType.Water_Level),
        },
    )
    assert ds.time[0].year == 2001
    assert ds["flow"].item.type == EUMType.Discharge
    assert ds["level"].item.name == "level"


def test_from_polars_use_first_datetime_column() -> None:
    import polars as pl

    df = pl.DataFrame(
        {
            "time": [
                datetime(2001, 1, 1, 0),
                datetime(2001, 1, 1, 1),
                datetime(2001, 1, 1, 2),
            ],
            "flow": [1.0, None, 2.0],
            "level": [2.0, 3.0, -1.3],
        }
    )

    ds = mikeio.from_polars(df)

    assert ds.n_timesteps == 3
    assert ds.time[-1].year == 2001
    assert ds["flow"].values[-1] == pytest.approx(2.0)


def test_write_from_named_pandas_series(tmp_path: Path) -> None:
    series = pd.Series(
        [1.0, 2.0, np.nan, np.nan],
        name="Average",
        index=pd.date_range("1958-01-01", periods=4, freq="D"),
    )

    filename = tmp_path / "series.dfs0"

    mikeio.from_pandas(
        series,
        items=mikeio.ItemInfo(EUMType.Concentration, EUMUnit.gram_per_meter_pow_3),
    ).to_dfs(
        filename=filename,
    )

    ds = mikeio.read(filename)

    assert len(ds.items) == 1
    assert ds[0].type == EUMType.Concentration
    assert ds[0].unit == EUMUnit.gram_per_meter_pow_3
    assert np.isnan(ds["Average"].to_numpy()[3])
    assert ds.time[0].year == 1958

    # Deprecated
    with pytest.warns(FutureWarning, match="from_pandas"):
        series.to_dfs0(
            filename, itemtype=EUMType.Concentration, unit=EUMUnit.gram_per_meter_pow_3
        )


def test_write_from_data_frame_different_types(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {"Average": [1.0, 2.0, np.nan], "Trend": [0.1, 0.2, 0.3]},
        index=pd.date_range("2000-01-01", periods=3, freq="D"),
    )

    filename = tmp_path / "dataframe.dfs0"

    items = [
        ItemInfo("Average", EUMType.Concentration, EUMUnit.gram_per_meter_pow_3),
        ItemInfo("Trend", EUMType.Undefined),
    ]

    mikeio.from_pandas(df, items=items).to_dfs(filename)

    ds = mikeio.read(filename)

    assert len(ds.items) == 2
    assert ds.items[0].type == EUMType.Concentration
    assert ds.items[0].unit == EUMUnit.gram_per_meter_pow_3

    assert ds.items[1].type == EUMType.Undefined
    assert ds.items[1].unit == EUMUnit.undefined


def test_read_dfs0_single_item() -> None:
    dfs0file = r"tests/testdata/random.dfs0"

    dfs = Dfs0(dfs0file)
    ds = dfs.read(items=[1])

    assert len(ds.to_numpy()) == 1


def test_read_dfs0_single_item_named_access() -> None:
    dfs0file = r"tests/testdata/random.dfs0"

    dfs = Dfs0(dfs0file)
    res = dfs.read(items=[1])

    assert len(res.to_numpy()) == 1


def test_read_dfs0_temporal_subset() -> None:
    dfs0file = r"tests/testdata/random.dfs0"

    dfs = Dfs0(dfs0file)
    ds = dfs.read(time=[1, 2])

    assert len(ds.time) == 2
    assert ds.time[0].strftime("%H") == "05"


def test_read_non_eq_dfs0_temporal_subset() -> None:
    dfs0file = r"tests/testdata/da_diagnostic.dfs0"

    dfs = Dfs0(dfs0file)

    ds = dfs.read(time=slice("2017-10-27 01:00", "2017-10-27 02:00"))

    assert len(ds.time) == 7


def test_read_non_eq_dfs0_temporal_slice() -> None:
    dfs0file = r"tests/testdata/da_diagnostic.dfs0"

    dfs = Dfs0(dfs0file)
    start = "2017-10-27 01:00"
    end = "2017-10-27 02:00"
    ds = dfs.read(time=slice(start, end))
    assert ds.time[0].strftime("%Y-%m-%d %H:%M") == start
    assert ds.time[-1].strftime("%Y-%m-%d %H:%M") == end
    assert len(ds.time) == 7


def test_read_dfs0_single_item_read_by_name() -> None:
    dfs0file = r"tests/testdata/random.dfs0"

    items = ["NotFun", "VarFun01"]
    dfs = Dfs0(dfs0file)
    res = dfs.read(items=items)  # reversed order compare to original file
    data = res.to_numpy()

    assert items == ["NotFun", "VarFun01"]
    assert len(data) == 2
    assert res.items[0].name == "NotFun"
    assert res.items[0].type == EUMType.Water_Level
    assert res.items[0].unit == EUMUnit.meter
    assert repr(res.items[0].unit) == "meter"


def test_read_dfs0_to_dataframe() -> None:
    dfs0file = r"tests/testdata/random.dfs0"

    dfs = Dfs0(dfs0file)
    df = dfs.to_dataframe()

    assert np.isnan(df[df.columns[0]].iloc[2])


def test_read_dfs0_to_matrix() -> None:
    dfs0file = r"tests/testdata/random.dfs0"

    dfs = Dfs0(dfs0file)
    ds = dfs.read()

    assert len(ds.to_numpy()) == 2


def test_write_data_with_missing_values(tmp_path: Path) -> None:
    dfs0file = r"tests/testdata/random.dfs0"
    tmpfile = tmp_path / "random.dfs0"

    ds = mikeio.read(dfs0file)

    # Do something with the data
    ds[0].values = np.zeros_like(ds[0].values)
    ds[1].values = np.ones_like(ds[0].values)

    # Add some NaNs
    ds[1].values[0:10] = np.nan

    # Overwrite the file
    ds.to_dfs(tmpfile)

    # Write operation does not modify the data
    assert np.isnan(ds[1].values[1])

    modified = mikeio.read(tmpfile)
    assert np.isnan(modified[1].values[5])


def test_read_relative_time_axis() -> None:
    filename = "tests/testdata/eq_relative.dfs0"

    ds = mikeio.read(filename)
    assert isinstance(ds.time, pd.TimedeltaIndex)
    assert ds.time[0].total_seconds() == 0.0
    assert ds.time[-1].total_seconds() == pytest.approx(56.236909)
    assert len(ds) == 5


def test_write_accumulated_datatype(tmp_path: Path) -> None:
    filename = tmp_path / "simple.dfs0"

    da = mikeio.DataArray(
        data=np.random.random(100),
        time=pd.date_range("2012-01-01", periods=100, freq="h"),
        item=ItemInfo(
            name="testing water level",
            itemtype=EUMType.Water_Level,
            unit=EUMUnit.meter,
            data_value_type="MeanStepBackward",
        ),
    )
    da.to_dfs(filename)

    da.to_dfs(filename)
    newds = mikeio.read(filename)
    assert newds[0].item.data_value_type == 3


def test_write_default_datatype(tmp_path: Path) -> None:
    filename = tmp_path / "simple.dfs0"
    da = mikeio.DataArray(
        data=np.random.random(100),
        time=pd.date_range("2012-01-01", periods=100, freq="h"),
        item=ItemInfo(
            name="testing water level",
            itemtype=EUMType.Water_Level,
            unit=EUMUnit.meter,
        ),
    )
    da.to_dfs(filename)
    newds = mikeio.read(filename)
    assert newds[0].item.data_value_type == 0


def test_write_from_pandas_series_data_value_not_default(tmp_path: Path) -> None:
    series = pd.Series(
        [1.0, 2.0, np.nan, np.nan],
        name="Average",
        index=pd.date_range("1958-01-01", periods=4, freq="D"),
    )

    filename = tmp_path / "series.dfs0"

    mikeio.from_pandas(
        series,
        items=[
            ItemInfo(
                EUMType.Concentration,
                EUMUnit.gram_per_meter_pow_3,
                data_value_type="MeanStepBackward",
            )
        ],
    ).to_dfs(filename)

    # TODO skip writing to disk?

    ds = mikeio.read(filename)

    assert len(ds.items) == 1
    assert ds[0].type == EUMType.Concentration
    assert ds[0].unit == EUMUnit.gram_per_meter_pow_3
    assert np.isnan(ds["Average"].to_numpy()[3])
    assert ds.time[0].year == 1958
    assert ds.items[0].data_value_type == DataValueType.MeanStepBackward


def test_write_from_data_frame_data_value_not_default(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {"Average": [1.0, 2.0, np.nan], "Trend": [0.1, 0.2, 0.3]},
        index=pd.date_range("1958-01-01", periods=3, freq="D"),
    )

    filename = tmp_path / "dataframe.dfs0"

    items = [
        ItemInfo(
            col,
            EUMType.Concentration,
            EUMUnit.gram_per_meter_pow_3,
            data_value_type="MeanStepBackward",
        )
        for col in df.columns
    ]

    mikeio.from_pandas(df, items=items).to_dfs(filename)

    ds = mikeio.read(filename)
    assert all(
        [item.data_value_type == DataValueType.MeanStepBackward for item in ds.items]
    )


def test_read_write_eum(tmp_path: Path) -> None:
    ds = mikeio.read("tests/testdata/waterlevel_viken.dfs0")
    assert ds["ST 2: WL (m)"].type == EUMType.Water_Level
    assert ds["ST 2: WL (m)"].unit == EUMUnit.meter

    outfilename = tmp_path / "same_same.dfs0"

    ds.to_dfs(outfilename)

    ds2 = mikeio.read(outfilename)
    assert ds2["ST 2: WL (m)"].type == EUMType.Water_Level
    assert ds2["ST 2: WL (m)"].unit == EUMUnit.meter


def test_read_write_single_step(tmp_path: Path) -> None:
    ds = mikeio.read("tests/testdata/waterlevel_viken.dfs0", time=-1)
    outfilename = tmp_path / "last_step.dfs0"
    ds.to_dfs(outfilename)

    dsnew = mikeio.read(outfilename)
    assert dsnew.n_timesteps == 1
    assert dsnew[0].to_numpy() == pytest.approx(-0.08139999955892563)


def test_read_write_single_step_to_dataframe(tmp_path: Path) -> None:
    ds = mikeio.read("tests/testdata/da_diagnostic.dfs0", time=1)
    df = ds.to_dataframe()
    assert df.shape[0] == 1
    assert df.iloc[0, 0] == pytest.approx(1.81134)
    assert np.isnan(df.iloc[0, 3])


def test_read_dfs0_with_many_items() -> None:
    ds = mikeio.read("tests/testdata/many_items.dfs0")

    assert ds.n_items == 800


def test_read_dfs0_with_non_unique_item_names() -> None:
    with pytest.warns(match="item name"):
        ds = mikeio.read("tests/testdata/untitled_3_items.dfs0")

    assert ds.n_items == 3

    assert ds["Untitled"].values[0] == pytest.approx(1.0)

    assert ds["Untitled_3"].values[0] == pytest.approx(0.0)
    assert np.isnan(ds.Untitled_3.values[1])  # type: ignore


def test_non_equidistant_time_can_read_correctly_with_open() -> None:
    dfs = mikeio.Dfs0("tests/testdata/neq_daily_time_unit.dfs0")
    ds = dfs.read()

    assert all(dfs.time == ds.time)


def test_temporal_selection_neq_time() -> None:
    dfs = mikeio.Dfs0("tests/testdata/Sirius_IDF_rainfall.dfs0")
    ds1 = dfs.read(time=[0, 1])
    assert ds1.n_timesteps == 2

    ds2 = dfs.read(time=slice(0, 2))
    assert ds2.n_timesteps == 2

    ds3 = dfs.read(time="2019-01-01 12:00:00")
    assert ds3.n_timesteps == 1

    assert dfs.n_timesteps == 22
    with pytest.raises(IndexError):
        ds1 = dfs.read(time=[0, 23])
