from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import pytest

import mikeio
from mikeio import EUMType, ItemInfo, EUMUnit, Dataset
from mikeio.exceptions import OutsideModelDomainError


@pytest.fixture
def ds1() -> Dataset:
    nt = 10
    ne = 7

    d1 = np.zeros([nt, ne]) + 0.1
    d2 = np.zeros([nt, ne]) + 0.2
    data = [d1, d2]

    time = pd.date_range(start=datetime(2000, 1, 1), freq="s", periods=nt)
    items = [ItemInfo("Foo"), ItemInfo("Bar")]
    return mikeio.Dataset.from_numpy(
        data=data, time=time, items=items, geometry=mikeio.Grid1D(nx=7, dx=1)
    )


@pytest.fixture
def ds2() -> Dataset:
    nt = 10
    ne = 7

    d1 = np.zeros([nt, ne]) + 1.0
    d2 = np.zeros([nt, ne]) + 2.0
    data = [d1, d2]

    time = pd.date_range(start=datetime(2000, 1, 1), freq="s", periods=nt)
    items = [ItemInfo("Foo"), ItemInfo("Bar")]
    return mikeio.Dataset.from_numpy(data=data, time=time, items=items)


@pytest.fixture
def ds3() -> Dataset:
    nt = 100
    d1 = np.zeros([nt, 100, 30]) + 1.5
    d2 = np.zeros([nt, 100, 30]) + 2.0
    d3 = np.zeros([nt, 100, 30]) + 3.0

    data = [d1, d2, d3]

    time = pd.date_range("2000-1-2", freq="h", periods=nt)
    items = [ItemInfo(x) for x in ["Foo", "Bar", "Baz"]]
    return mikeio.Dataset.from_numpy(data=data, time=time, items=items)


def test_list_of_numpy_deprecated() -> None:
    with pytest.warns(FutureWarning, match="from_numpy"):
        mikeio.Dataset([np.zeros(2)], pd.date_range("2000-1-2", freq="h", periods=2))  # type: ignore


def test_get_names() -> None:
    nt = 100
    d = np.zeros([nt, 100, 30]) + 1.0
    time = pd.date_range(start=datetime(2000, 1, 1), freq="s", periods=nt)
    data_vars = {"Foo": mikeio.DataArray(data=d, time=time, item=ItemInfo(name="Foo"))}

    ds = mikeio.Dataset(data_vars)

    assert ds["Foo"].name == "Foo"
    assert ds["Foo"].type == EUMType.Undefined
    assert repr(ds["Foo"].unit) == "undefined"
    assert ds.names == ["Foo"]


def test_properties(ds1: Dataset) -> None:
    nt = 10
    ne = 7
    time = pd.date_range(start=datetime(2000, 1, 1), freq="s", periods=nt)

    assert ds1.names == ["Foo", "Bar"]
    assert ds1.n_items == 2

    assert np.all(ds1.time == time)
    assert ds1.n_timesteps == nt
    assert ds1.timestep == 1
    assert ds1.start_time == time[0]
    assert ds1.end_time == time[-1]

    assert ds1.shape == (nt, ne)
    assert ds1.dims == ("time", "x")
    assert ds1.geometry.nx == 7
    assert ds1._zn is None

    # assert not hasattr(ds1, "keys")   # TODO: remove this
    # assert not hasattr(ds1, "values") # TODO: remove this
    assert isinstance(ds1.items[0], ItemInfo)


def test_insert(ds1: Dataset) -> None:
    da = ds1[0].copy()
    da.name = "Baz"

    ds1.insert(2, da)
    assert len(ds1) == 3
    assert ds1.names == ["Foo", "Bar", "Baz"]
    assert ds1[-1] == da


def test_remove(ds1: Dataset) -> None:
    ds1.remove(-1)
    assert len(ds1) == 1
    assert ds1.names == ["Foo"]

    ds1.remove("Foo")
    with pytest.raises(KeyError):
        ds1.remove("Foo")
    assert len(ds1) == 0


def test_index_with_attribute() -> None:
    nt = 10
    d = np.zeros([nt, 100, 30]) + 1.0
    time = pd.date_range(start=datetime(2000, 1, 1), freq="s", periods=nt)

    da1 = mikeio.DataArray(name="Foo", data=d, time=time)
    da2 = mikeio.DataArray(name="Bar", data=d.copy(), time=time)
    data = [da1, da2]
    ds = mikeio.Dataset(data)
    assert ds["Foo"].name == "Foo"
    assert ds.Bar.name == "Bar"  # type: ignore

    assert ds["Foo"] is ds.Foo  # type: ignore

    ds["Foo"] = ds.Foo + 2.0  # type: ignore
    assert (
        ds["Foo"] is ds.Foo  # type: ignore
    )  # This is now modfied, but both methods points to the same object


def test_getitem_time(ds3: Dataset) -> None:
    # time = pd.date_range("2000-1-2", freq="h", periods=100)

    # deprecated use .sel(time=...) or .isel(time=...) instead
    with pytest.warns(FutureWarning, match="time"):
        ds_sel = ds3["2000-1-2"]  # type: ignore
    assert ds_sel.n_timesteps == 24
    assert ds_sel.is_equidistant

    with pytest.warns(FutureWarning, match="time"):
        ds_sel = ds3["2000-1-2":"2000-1-3 00:00"]  # type: ignore
    assert ds_sel.n_timesteps == 25
    assert ds_sel.is_equidistant

    with pytest.warns(FutureWarning, match="time"):
        time = ["2000-1-2 04:00:00", "2000-1-2 08:00:00", "2000-1-2 12:00:00"]
        ds_sel = ds3[time]  # type: ignore
    assert ds_sel.n_timesteps == 3
    assert ds_sel.is_equidistant

    with pytest.warns(FutureWarning, match="time"):
        time = [ds3.time[0], ds3.time[1], ds3.time[7], ds3.time[23]]
        ds_sel = ds3[time]  # type: ignore
    assert ds_sel.n_timesteps == 4
    assert not ds_sel.is_equidistant

    with pytest.warns(FutureWarning, match="time"):
        ds_sel = ds3[ds3.time[:10]]  # type: ignore
    assert ds_sel.n_timesteps == 10
    assert ds_sel.is_equidistant


def test_getitem_multi_indexing_attempted(ds3: Dataset) -> None:
    with pytest.raises(TypeError, match="not allow multi-index"):
        ds3[0, 0]
    with pytest.warns(Warning, match="ambiguity"):
        ds3[0, 1]  # indistinguishable from ds3[(0,1)]
    with pytest.raises(TypeError, match="not allow multi-index"):
        ds3[:, 1]
    with pytest.raises(TypeError, match="not allow multi-index"):
        ds3[-1, [0, 1], 1]


def test_select_subset_isel() -> None:
    nt = 100
    d1 = np.zeros([nt, 100, 30]) + 1.5
    d2 = np.zeros([nt, 100, 30]) + 2.0

    d1[0, 10, :] = 2.0
    d2[0, 10, :] = 3.0

    time = pd.date_range(start=datetime(2000, 1, 1), freq="s", periods=nt)

    geometry = mikeio.Grid2D(nx=30, ny=100, bbox=[0, 0, 1, 1])

    data = {
        "Foo": mikeio.DataArray(
            data=d1, time=time, geometry=geometry, item=ItemInfo("Foo")
        ),
        "Bar": mikeio.DataArray(
            data=d2, time=time, geometry=geometry, item=ItemInfo("Bar")
        ),
    }

    ds = mikeio.Dataset(data)

    selds = ds.isel(y=10)

    assert len(selds.items) == 2
    assert len(selds.to_numpy()) == 2
    assert selds["Foo"].shape == (100, 30)
    assert selds["Foo"].to_numpy()[0, 0] == 2.0
    assert selds["Bar"].to_numpy()[0, 0] == 3.0


def test_select_subset_isel_axis_out_of_range_error(ds2: Dataset) -> None:
    assert len(ds2.shape) == 2
    dss = ds2.isel(idx=0)

    # After subsetting there is only one dimension
    assert "y" not in dss.dims

    with pytest.raises(ValueError):
        dss.isel(y=0)


def test_isel_named_axis(ds2: mikeio.Dataset) -> None:
    dss = ds2.isel(time=0)
    assert len(dss.shape) == 1


def test_select_temporal_subset_by_idx() -> None:
    nt = 100
    d1 = np.zeros([nt, 100, 30]) + 1.5
    d2 = np.zeros([nt, 100, 30]) + 2.0

    d1[0, 10, :] = 2.0
    d2[0, 10, :] = 3.0
    data = [d1, d2]

    time = pd.date_range(start=datetime(2000, 1, 1), freq="s", periods=nt)
    items = [ItemInfo("Foo"), ItemInfo("Bar")]
    ds = mikeio.Dataset.from_numpy(data=data, time=time, items=items)

    selds = ds.isel(time=[0, 1, 2])

    assert len(selds) == 2
    assert selds["Foo"].shape == (3, 100, 30)


def test_temporal_subset_fancy() -> None:
    # TODO use .sel(time=...) instead, more explicit
    nt = (24 * 31) + 1
    d1 = np.zeros([nt, 100, 30]) + 1.5
    d2 = np.zeros([nt, 100, 30]) + 2.0
    data = [d1, d2]

    time = pd.date_range("2000-1-1", freq="h", periods=nt)
    items = [ItemInfo("Foo"), ItemInfo("Bar")]
    ds = mikeio.Dataset.from_numpy(data=data, time=time, items=items)

    assert ds.time[0].hour == 0
    assert ds.time[-1].hour == 0

    with pytest.warns(FutureWarning, match="time"):
        selds = ds["2000-01-01 00:00":"2000-01-02 00:00"]  # type: ignore

    assert len(selds) == 2
    assert selds["Foo"].shape == (25, 100, 30)

    with pytest.warns(FutureWarning, match="time"):
        selds = ds[:"2000-01-02 00:00"]  # type: ignore
    assert selds["Foo"].shape == (25, 100, 30)

    with pytest.warns(FutureWarning, match="time"):
        selds = ds["2000-01-31 00:00":]  # type: ignore
    assert selds["Foo"].shape == (25, 100, 30)

    with pytest.warns(FutureWarning, match="time"):
        selds = ds["2000-01-30":]  # type: ignore
    assert selds["Foo"].shape == (49, 100, 30)


def test_select_item_by_name() -> None:
    nt = 100
    d1 = np.zeros([nt, 100, 30]) + 1.5
    d2 = np.zeros([nt, 100, 30]) + 2.0

    d1[0, 10, :] = 2.0
    d2[0, 10, :] = 3.0
    data = [d1, d2]

    time = pd.date_range("2000-1-2", freq="h", periods=nt)
    items = [ItemInfo("Foo"), ItemInfo("Bar")]
    ds = mikeio.Dataset.from_numpy(data=data, time=time, items=items)

    foo_data = ds["Foo"]
    assert foo_data.to_numpy()[0, 10, 0] == 2.0


def test_missing_item_error() -> None:
    nt = 100

    da1 = mikeio.DataArray(
        data=np.zeros(nt),
        time=pd.date_range("2000-1-2", freq="h", periods=nt),
        name="Foo",
    )

    da2 = mikeio.DataArray(
        data=np.ones(nt),
        time=pd.date_range("2000-1-2", freq="h", periods=nt),
        name="Bar",
    )

    ds = mikeio.Dataset([da1, da2])

    with pytest.raises(KeyError, match="Baz"):
        ds["Baz"]  # there is no Bar item


def test_select_multiple_items_by_name() -> None:
    nt = 100
    d1 = np.zeros([nt, 100, 30]) + 1.5
    d2 = np.zeros([nt, 100, 30]) + 2.0
    d3 = np.zeros([nt, 100, 30]) + 3.0

    data = [d1, d2, d3]

    time = pd.date_range("2000-1-2", freq="h", periods=nt)
    # items = [ItemInfo("Foo"), ItemInfo("Bar"), ItemInfo("Baz")]
    items = [ItemInfo(x) for x in ["Foo", "Bar", "Baz"]]
    ds = mikeio.Dataset.from_numpy(data=data, time=time, items=items)

    assert len(ds) == 3  # Length of a dataset is the number of items

    newds = ds[["Baz", "Foo"]]
    assert newds.items[0].name == "Baz"
    assert newds.items[1].name == "Foo"
    assert newds["Foo"].to_numpy()[0, 10, 0] == 1.5

    assert len(newds) == 2


def test_select_multiple_items_by_index(ds3: Dataset) -> None:
    assert len(ds3) == 3  # Length of a dataset is the number of items

    newds = ds3[[2, 0]]
    assert len(newds) == 2
    assert newds.items[0].name == "Baz"
    assert newds.items[1].name == "Foo"
    assert newds["Foo"].to_numpy()[0, 10, 0] == 1.5


def test_select_multiple_items_by_slice(ds3: Dataset) -> None:
    assert len(ds3) == 3  # Length of a dataset is the number of items

    newds = ds3[:2]
    assert len(newds) == 2
    assert newds.items[0].name == "Foo"
    assert newds.items[1].name == "Bar"
    assert newds["Foo"].to_numpy()[0, 10, 0] == 1.5


def test_select_item_by_iteminfo() -> None:
    nt = 100
    d1 = np.zeros([nt, 100, 30]) + 1.5
    d2 = np.zeros([nt, 100, 30]) + 2.0

    d1[0, 10, :] = 2.0
    d2[0, 10, :] = 3.0
    data = [d1, d2]

    time = pd.date_range("2000-1-2", freq="h", periods=nt)
    items = [ItemInfo("Foo"), ItemInfo("Bar")]
    ds = mikeio.Dataset.from_numpy(data=data, time=time, items=items)

    foo_item = items[0]

    foo_data = ds[foo_item]
    assert foo_data.to_numpy()[0, 10, 0] == 2.0


def test_select_subset_isel_multiple_idxs() -> None:
    nt = 100
    d1 = np.zeros([nt, 100, 30]) + 1.5
    d2 = np.zeros([nt, 100, 30]) + 2.0

    data = [d1, d2]

    time = pd.date_range("2000-1-2", freq="h", periods=nt)
    items = [ItemInfo("Foo"), ItemInfo("Bar")]
    ds = mikeio.Dataset.from_numpy(data=data, time=time, items=items)

    selds = ds.isel(y=[10, 15])

    assert len(selds.items) == 2
    assert len(selds.to_numpy()) == 2
    assert selds["Foo"].shape == (100, 2, 30)


def test_decribe(ds1: Dataset) -> None:
    df = ds1.describe()
    assert df.columns[0] == "Foo"
    assert df.Bar["mean"] == pytest.approx(0.2)
    assert df.Foo["max"] == pytest.approx(0.1)


def test_create_undefined() -> None:
    nt = 100
    d1 = np.zeros([nt])
    d2 = np.zeros([nt])

    time = pd.date_range("2000-1-2", freq="h", periods=nt)
    data = {
        "Item 1": mikeio.DataArray(
            data=d1, time=time, item=ItemInfo("Item 1")
        ),  # TODO redundant name
        "Item 2": mikeio.DataArray(data=d2, time=time, item=ItemInfo("Item 2")),
    }

    ds = mikeio.Dataset(data)

    assert len(ds.items) == 2
    assert len(ds.to_numpy()) == 2
    assert ds[0].name == "Item 1"
    assert ds[0].type == EUMType.Undefined


def test_to_dataframe_single_timestep() -> None:
    nt = 1
    d1 = np.zeros([nt])
    d2 = np.zeros([nt])

    data = [d1, d2]

    time = pd.date_range("2000-1-2", freq="h", periods=nt)
    items = [ItemInfo("Foo"), ItemInfo("Bar")]
    ds = mikeio.Dataset.from_numpy(data=data, time=time, items=items)
    df = ds.to_dataframe()

    assert "Bar" in df.columns
    # assert isinstance(df.index, pd.DatetimeIndex)

    df2 = ds.to_pandas()
    assert df2.shape == (1, 2)


def test_to_dataframe() -> None:
    nt = 100
    d1 = np.zeros([nt])
    d2 = np.zeros([nt])

    data = [d1, d2]

    time = pd.date_range("2000-1-2", freq="h", periods=nt)
    items = [ItemInfo("Foo"), ItemInfo("Bar")]
    ds = mikeio.Dataset.from_numpy(data=data, time=time, items=items)
    df = ds.to_dataframe()

    assert list(df.columns) == ["Foo", "Bar"]
    assert isinstance(df.index, pd.DatetimeIndex)


def test_to_pandas_single_item_dataset() -> None:
    da = mikeio.DataArray(
        data=np.zeros(5), time=pd.date_range("2000", freq="D", periods=5), name="Foo"
    )
    ds = mikeio.Dataset([da])

    series = ds.to_pandas()

    assert series.shape == (5,)
    assert series.name == "Foo"


def test_multidimensional_to_dataframe_no_supported() -> None:
    nt = 100
    d1 = np.zeros([nt, 2])

    time = pd.date_range("2000-1-2", freq="h", periods=nt)
    items = [ItemInfo("Foo")]
    ds = mikeio.Dataset.from_numpy(data=[d1], time=time, items=items)

    with pytest.raises(ValueError):
        ds.to_dataframe()


def test_get_data() -> None:
    data = []
    nt = 100
    d = np.zeros([nt, 100, 30]) + 1.0
    data.append(d)
    time = pd.date_range("2000-1-2", freq="h", periods=nt)
    items = [ItemInfo("Foo")]
    ds = mikeio.Dataset.from_numpy(data=data, time=time, items=items)

    assert ds.shape == (100, 100, 30)


def test_interp_time() -> None:
    nt = 4
    d = np.zeros([nt, 10, 3])
    d[1] = 2.0
    d[3] = 4.0
    data = [d]
    time = pd.date_range("2000-1-1", freq="d", periods=nt)
    items = [ItemInfo("Foo")]
    ds = mikeio.Dataset.from_numpy(data=data, time=time, items=items)

    assert ds[0].shape == (nt, 10, 3)

    dsi = ds.interp_time(dt=3600)

    assert ds.time[0] == dsi.time[0]
    assert dsi[0].shape == (73, 10, 3)

    dsi2 = ds.interp_time(freq="2h")
    assert dsi2.timestep == 2 * 3600

    with pytest.raises(ValueError, match="dt or freq"):
        ds.interp_time()

    dsi3 = ds.interp(time=pd.date_range("2000-1-1", freq="h", periods=10))
    assert dsi3.time[0] == pd.Timestamp("2000-01-01 00:00:00")
    assert dsi3.time[-1] == pd.Timestamp("2000-01-01 09:00:00")


def test_interp_time_to_other_dataset() -> None:
    # Arrange
    ## mikeio.Dataset 1
    nt = 4
    data = [np.zeros([nt, 10, 3])]
    time = pd.date_range("2000-1-1", freq="D", periods=nt)
    items = [ItemInfo("Foo")]
    ds1 = mikeio.Dataset.from_numpy(data=data, time=time, items=items)
    assert ds1.shape == (nt, 10, 3)

    ## mikeio.Dataset 2
    nt = 12
    data = [np.ones([nt, 10, 3])]
    time = pd.date_range("2000-1-1", freq="h", periods=nt)
    items = [ItemInfo("Foo")]
    ds2 = mikeio.Dataset.from_numpy(data=data, time=time, items=items)

    # Act
    ## interp
    dsi = ds1.interp_time(dt=ds2.time)

    # Assert
    assert dsi.time[0] == ds2.time[0]
    assert dsi.time[-1] == ds2.time[-1]
    assert len(dsi.time) == len(ds2.time)
    assert dsi[0].shape[0] == ds2[0].shape[0]

    # Accept dataset as argument
    dsi2 = ds1.interp_time(ds2)
    assert dsi2.time[0] == ds2.time[0]


def test_extrapolate() -> None:
    # Arrange
    ## mikeio.Dataset 1
    nt = 2
    data = [np.zeros([nt, 10, 3])]
    time = pd.date_range("2000-1-1", freq="D", periods=nt)
    items = [ItemInfo("Foo")]
    ds1 = mikeio.Dataset.from_numpy(data=data, time=time, items=items)
    assert ds1.shape == (nt, 10, 3)

    ## mikeio.Dataset 2 partly overlapping with mikeio.Dataset 1
    nt = 3
    data = [np.ones([nt, 10, 3])]
    time = pd.date_range("2000-1-2", freq="h", periods=nt)
    items = [ItemInfo("Foo")]
    ds2 = mikeio.Dataset.from_numpy(data=data, time=time, items=items)

    # Act
    ## interp
    dsi = ds1.interp_time(dt=ds2.time, fill_value=1.0)

    # Assert
    assert dsi.time[0] == ds2.time[0]
    assert dsi.time[-1] == ds2.time[-1]
    assert len(dsi.time) == len(ds2.time)
    assert dsi[0].values[0] == pytest.approx(0.0)
    assert dsi[0].values[1] == pytest.approx(1.0)  # filled
    assert dsi[0].values[2] == pytest.approx(1.0)  # filled


def test_extrapolate_not_allowed() -> None:
    ## mikeio.Dataset 1
    nt = 2
    data = [np.zeros([nt, 10, 3])]
    time = pd.date_range("2000-1-1", freq="D", periods=nt)
    items = [ItemInfo("Foo")]
    ds1 = mikeio.Dataset.from_numpy(data=data, time=time, items=items)
    assert ds1.shape == (nt, 10, 3)

    ## mikeio.Dataset 2 partly overlapping with mikeio.Dataset 1
    nt = 3
    data = [np.ones([nt, 10, 3])]
    time = pd.date_range("2000-1-2", freq="h", periods=nt)
    items = [ItemInfo("Foo")]
    ds2 = mikeio.Dataset.from_numpy(data=data, time=time, items=items)

    with pytest.raises(ValueError):
        ds1.interp_time(dt=ds2.time, fill_value=1.0, extrapolate=False)


def test_get_data_2() -> None:
    nt = 100
    data = []
    d = np.zeros([nt, 100, 30]) + 1.0
    data.append(d)
    time = pd.date_range("2000-1-2", freq="h", periods=nt)
    items = [ItemInfo("Foo")]
    mikeio.Dataset.from_numpy(data=data, time=time, items=items)

    assert data[0].shape == (100, 100, 30)


def test_get_data_name() -> None:
    nt = 100
    data = []
    d = np.zeros([nt, 100, 30]) + 1.0
    data.append(d)
    time = pd.date_range("2000-1-2", freq="h", periods=nt)
    items = [ItemInfo("Foo")]
    ds = mikeio.Dataset.from_numpy(data=data, time=time, items=items)

    assert ds["Foo"].shape == (100, 100, 30)


def test_modify_selected_variable() -> None:
    nt = 100

    time = pd.date_range("2000-1-2", freq="h", periods=nt)
    items = [ItemInfo("Foo")]
    ds = mikeio.Dataset.from_numpy([np.zeros((nt, 10))], time, items)

    assert ds.Foo.to_numpy()[0, 0] == 0.0  # type: ignore

    foo = ds.Foo  # type: ignore
    foo_mod = foo + 1.0

    ds["Foo"] = foo_mod
    assert ds.Foo.to_numpy()[0, 0] == 1.0  # type: ignore


def test_get_bad_name() -> None:
    nt = 100
    data = []
    d = np.zeros([100, 100, 30]) + 1.0
    data.append(d)
    time = pd.date_range("2000-1-2", freq="h", periods=nt)
    items = [ItemInfo("Foo")]
    ds = mikeio.Dataset.from_numpy(data=data, time=time, items=items)

    with pytest.raises(Exception):
        ds["BAR"]


def test_flipud() -> None:
    nt = 2
    d = np.random.random([nt, 100, 30])
    time = pd.date_range("2000-1-2", freq="h", periods=nt)
    items = [ItemInfo("Foo")]
    ds = mikeio.Dataset.from_numpy([d], time, items)

    dsud = ds.copy()
    dsud.flipud()

    assert dsud.shape == ds.shape
    assert dsud["Foo"].to_numpy()[0, 0, 0] == ds["Foo"].to_numpy()[0, -1, 0]


def test_aggregation_workflows(tmp_path: Path) -> None:
    # TODO move to integration tests
    filename = "tests/testdata/HD2D.dfsu"
    dfs = mikeio.Dfsu2DH(filename)

    ds = dfs.read(items=["Surface elevation", "Current speed"])
    ds2 = ds.max(axis=1)

    outfilename = tmp_path / "max.dfs0"
    ds2.to_dfs(outfilename)
    assert outfilename.exists()

    ds3 = ds.min(axis=1)

    outfilename = tmp_path / "min.dfs0"
    ds3.to_dfs(outfilename)
    assert outfilename.exists()


def test_aggregation_dataset_no_time() -> None:
    filename = "tests/testdata/HD2D.dfsu"
    dfs = mikeio.Dfsu2DH(filename)
    ds = dfs.read(time=-1, items=["Surface elevation", "Current speed"])

    ds2 = ds.max()
    assert ds2["Current speed"].values == pytest.approx(1.6463733)


def test_aggregations() -> None:
    filename = "tests/testdata/gebco_sound.dfs2"
    ds = mikeio.read(filename)

    for axis in [0, 1, 2]:
        ds.mean(axis=axis)
        ds.nanmean(axis=axis)
        ds.nanmin(axis=axis)
        ds.nanmax(axis=axis)

    assert ds.mean().shape == (264, 216)
    assert ds.mean(axis="time").shape == (264, 216)
    assert ds.mean(axis="spatial").shape == (1,)
    assert ds.mean(axis="space").shape == (1,)

    with pytest.raises(ValueError, match="space"):
        ds.mean(axis="spaghetti")

    dsm = ds.mean(axis="time")
    assert dsm.geometry is not None


def test_to_dfs_extension_validation(tmp_path: Path) -> None:
    outfilename = tmp_path / "not_gonna_happen.dfs2"

    ds = mikeio.read(
        "tests/testdata/HD2D.dfsu", items=["Surface elevation", "Current speed"]
    )
    with pytest.raises(ValueError) as excinfo:
        ds.to_dfs(outfilename)

    assert "dfsu" in str(excinfo.value)


def test_weighted_average(tmp_path: Path) -> None:
    # TODO move to integration tests
    fp = Path("tests/testdata/HD2D.dfsu")
    dfs = mikeio.Dfsu2DH(fp)

    ds = dfs.read(items=["Surface elevation", "Current speed"])

    area = dfs.geometry.get_element_area()
    ds2 = ds.average(weights=area, axis=1)

    out_path = tmp_path / "average.dfs0"
    ds2.to_dfs(out_path)
    assert out_path.exists()


def test_quantile_axis1(ds1: Dataset) -> None:
    dsq = ds1.quantile(q=0.345, axis=1)
    assert dsq[0].to_numpy()[0] == 0.1
    assert dsq[1].to_numpy()[0] == 0.2

    assert dsq.n_items == ds1.n_items
    assert dsq.n_timesteps == ds1.n_timesteps

    # q as list
    dsq = ds1.quantile(q=[0.25, 0.75], axis=1)
    assert dsq.n_items == 2 * ds1.n_items
    assert "Quantile 0.75, " in dsq.items[1].name
    assert "Quantile 0.25, " in dsq.items[2].name


def test_quantile_axis0(ds1: Dataset) -> None:
    dsq = ds1.quantile(q=0.345)  # axis=0 is default
    assert dsq[0].to_numpy()[0] == 0.1
    assert dsq[1].to_numpy()[0] == 0.2

    assert dsq.n_items == ds1.n_items
    assert dsq.n_timesteps == 1
    assert dsq.shape[-1] == ds1.shape[-1]

    # q as list
    dsq = ds1.quantile(q=[0.25, 0.75], axis=0)
    assert dsq.n_items == 2 * ds1.n_items
    assert dsq[0].to_numpy()[0] == 0.1
    assert dsq[1].to_numpy()[0] == 0.1
    assert dsq[2].to_numpy()[0] == 0.2
    assert dsq[3].to_numpy()[0] == 0.2

    assert "Quantile 0.75, " in dsq.items[1].name
    assert "Quantile 0.25, " in dsq.items[2].name
    assert "Quantile 0.75, " in dsq.items[3].name


def test_nanquantile() -> None:
    q = 0.99
    fn = "tests/testdata/random.dfs0"  # has delete value
    ds = mikeio.read(fn)

    dsq1 = ds.quantile(q=q)
    dsq2 = ds.nanquantile(q=q)

    assert np.isnan(dsq1[0].to_numpy())
    assert not np.isnan(dsq2[0].to_numpy())

    qnt = np.quantile(ds[0].to_numpy(), q=q)
    nqnt = np.nanquantile(ds[0].to_numpy(), q=q)

    assert np.isnan(qnt)
    assert dsq2[0].to_numpy() == nqnt


def test_aggregate_across_items() -> None:
    ds = mikeio.read("tests/testdata/State_wlbc_north_err.dfs1")

    dsm = ds.mean(axis="items")

    assert isinstance(dsm, mikeio.Dataset)
    assert dsm.geometry == ds.geometry
    assert dsm.dims == ds.dims

    dsq = ds.quantile(q=[0.1, 0.5, 0.9], axis="items")
    assert isinstance(dsq, mikeio.Dataset)
    assert dsq[0].name == "Quantile 0.1"
    assert dsq[1].name == "Quantile 0.5"
    assert dsq[2].name == "Quantile 0.9"

    # TODO allow name to be specified similar to below


def test_aggregate_selected_items_dfsu_save_to_new_file(tmp_path: Path) -> None:
    ds = mikeio.read("tests/testdata/State_Area.dfsu", items="*Level*")

    assert ds.n_items == 5

    dsm = ds.max(axis="items", name="Max Water Level")  # add a nice name
    assert len(dsm) == 1
    assert dsm[0].name == "Max Water Level"
    assert dsm.geometry == ds.geometry
    assert dsm.dims == ds.dims
    assert dsm[0].type == ds[-1].type

    outfilename = tmp_path / "maxwl.dfsu"
    dsm.to_dfs(outfilename)


def test_copy() -> None:
    nt = 100
    d1 = np.zeros([nt, 100, 30]) + 1.5
    d2 = np.zeros([nt, 100, 30]) + 2.0

    data = [d1, d2]

    time = pd.date_range("2000-1-2", freq="h", periods=nt)
    items = [ItemInfo("Foo"), ItemInfo("Bar")]
    ds = mikeio.Dataset.from_numpy(data=data, time=time, items=items)

    assert len(ds.items) == 2
    assert len(ds.to_numpy()) == 2
    assert ds[0].name == "Foo"

    ds2 = ds.copy()

    ds2[0].name = "New name"

    assert ds2[0].name == "New name"
    assert ds[0].name == "Foo"


def test_dropna() -> None:
    nt = 10
    d1 = np.zeros([nt, 100, 30])
    d2 = np.zeros([nt, 100, 30])

    d1[9:] = np.nan
    d2[8:] = np.nan

    data = [d1, d2]

    time = pd.date_range("2000-1-2", freq="h", periods=nt)
    items = [ItemInfo("Foo"), ItemInfo("Bar")]
    ds = mikeio.Dataset.from_numpy(data=data, time=time, items=items)

    assert len(ds.items) == 2
    assert len(ds.to_numpy()) == 2

    ds2 = ds.dropna()

    assert ds2.n_timesteps == 8


def test_default_type() -> None:
    item = ItemInfo("Foo")
    assert item.type == EUMType.Undefined
    assert repr(item.unit) == "undefined"


def test_int_is_valid_type_info() -> None:
    item = ItemInfo("Foo", 100123)
    assert item.type == EUMType.Viscosity

    item = ItemInfo("U", 100002)
    assert item.type == EUMType.Wind_Velocity


def test_int_is_valid_unit_info() -> None:
    item = ItemInfo("U", 100002, 2000)
    assert item.type == EUMType.Wind_Velocity
    assert item.unit == EUMUnit.meter_per_sec
    assert repr(item.unit) == "meter per sec"  # TODO replace _per_ with /


def test_default_unit_from_type() -> None:
    item = ItemInfo("Foo", EUMType.Water_Level)
    assert item.type == EUMType.Water_Level
    assert item.unit == EUMUnit.meter
    assert repr(item.unit) == "meter"

    item = ItemInfo("Tp", EUMType.Wave_period)
    assert item.type == EUMType.Wave_period
    assert item.unit == EUMUnit.second
    assert repr(item.unit) == "second"

    item = ItemInfo("Temperature", EUMType.Temperature)
    assert item.type == EUMType.Temperature
    assert item.unit == EUMUnit.degree_Celsius
    assert repr(item.unit) == "degree Celsius"


def test_default_name_from_type() -> None:
    item = ItemInfo(EUMType.Current_Speed)
    assert item.name == "Current Speed"
    assert item.unit == EUMUnit.meter_per_sec

    item2 = ItemInfo(EUMType.Current_Direction, EUMUnit.degree)
    assert item2.unit == EUMUnit.degree
    item3 = ItemInfo(
        "Current direction (going to)", EUMType.Current_Direction, EUMUnit.degree
    )
    assert item3.type == EUMType.Current_Direction
    assert item3.unit == EUMUnit.degree


def test_iteminfo_string_type_should_fail_with_helpful_message() -> None:
    with pytest.raises(ValueError):
        ItemInfo("Water level", "Water level")  # type: ignore


def test_item_search() -> None:
    res = EUMType.search("level")

    assert len(res) > 0
    assert isinstance(res[0], EUMType)


def test_dfsu3d_dataset() -> None:
    filename = "tests/testdata/oresund_sigma_z.dfsu"

    dfsu = mikeio.Dfsu3D(filename)

    ds = dfsu.read()

    repr(ds)

    assert len(ds) == 2  # Salinity, Temperature

    dsagg = ds.nanmean(axis=0)  # Time averaged

    assert len(dsagg) == 2  # Salinity, Temperature

    assert dsagg[0].shape[0] == 17118

    assert dsagg.time[0] == ds.time[0]  # Time-averaged data index by start time

    ds_elm = dfsu.read(elements=[0])

    assert len(ds_elm) == 2  # Salinity, Temperature

    dss = ds_elm.squeeze()

    assert len(dss) == 2  # Salinity, Temperature


def test_items_data_mismatch() -> None:
    nt = 100
    d = np.zeros([nt, 100, 30]) + 1.0
    time = pd.date_range("2000-1-2", freq="h", periods=nt)
    items = [ItemInfo("Foo"), ItemInfo("Bar")]  # Two items is not correct!

    with pytest.raises(ValueError):
        mikeio.Dataset.from_numpy([d], time, items)


def test_time_data_mismatch() -> None:
    nt = 100
    d = np.zeros([nt, 100, 30]) + 1.0
    time = pd.date_range(
        "2000-1-2", freq="h", periods=nt + 1
    )  # 101 timesteps is not correct!
    items = [ItemInfo("Foo")]

    with pytest.raises(ValueError):
        mikeio.Dataset.from_numpy([d], time, items)


def test_properties_dfs2() -> None:
    filename = "tests/testdata/gebco_sound.dfs2"
    ds = mikeio.read(filename)

    assert ds.n_timesteps == 1
    assert ds.n_items == 1
    assert np.all(ds.shape == (1, 264, 216))
    assert ds.n_elements == (264 * 216)
    assert ds.is_equidistant


def test_properties_dfsu() -> None:
    filename = "tests/testdata/oresund_vertical_slice.dfsu"
    ds = mikeio.read(filename)

    assert ds.n_timesteps == 3
    assert ds.start_time == datetime(1997, 9, 15, 21, 0, 0)
    assert ds.end_time == datetime(1997, 9, 16, 3, 0, 0)
    assert ds.timestep == (3 * 3600)
    assert ds.n_items == 2
    assert np.all(ds.shape == (3, 441))
    assert ds.n_elements == 441
    assert ds.is_equidistant


def test_create_empty_data() -> None:
    ne = 34
    d = mikeio.Dataset.create_empty_data(n_elements=ne)
    assert len(d) == 1
    assert d[0].shape == (1, ne)

    nt = 100
    d = mikeio.Dataset.create_empty_data(n_timesteps=nt, shape=(3, 4, 6))
    assert len(d) == 1
    assert d[0].shape == (nt, 3, 4, 6)

    ni = 4
    d = mikeio.Dataset.create_empty_data(n_items=ni, n_elements=ne)
    assert len(d) == ni
    assert d[-1].shape == (1, ne)

    with pytest.raises(Exception):
        d = mikeio.Dataset.create_empty_data()

    with pytest.raises(Exception):
        d = mikeio.Dataset.create_empty_data(n_elements=None, shape=None)


def test_create_infer_name_from_eum() -> None:
    nt = 100
    d = np.random.uniform(size=nt)

    ds = mikeio.Dataset.from_numpy(
        data=[d],
        time=pd.date_range("2000-01-01", freq="h", periods=nt),
        items=[ItemInfo(EUMType.Wind_speed)],
    )

    assert isinstance(ds.items[0], ItemInfo)
    assert ds.items[0].type == EUMType.Wind_speed
    assert ds.items[0].name == "Wind speed"


def test_add_scalar(ds1: Dataset) -> None:
    ds2 = ds1 + 10.0
    assert np.all(ds2[0].to_numpy() - ds1[0].to_numpy() == 10.0)

    ds3 = 10.0 + ds1
    assert np.all(ds3[0].to_numpy() == ds2[0].to_numpy())
    assert np.all(ds3[1].to_numpy() == ds2[1].to_numpy())


def test_add_inconsistent_dataset(ds1: Dataset) -> None:
    ds2 = ds1[[0]]

    assert len(ds1) != len(ds2)

    with pytest.raises(ValueError):
        ds1 + ds2

    with pytest.raises(ValueError):
        ds1 * ds2


def test_add_bad_value(ds1: Dataset) -> None:
    with pytest.raises(TypeError):
        ds1 + ["one"]  # type: ignore


def test_multiple_bad_value(ds1: Dataset) -> None:
    with pytest.raises(TypeError):
        ds1 * ["pi"]  # type: ignore


def test_sub_scalar(ds1: Dataset) -> None:
    ds2 = ds1 - 10.0
    assert isinstance(ds2, mikeio.Dataset)
    assert np.all(ds1[0].to_numpy() - ds2[0].to_numpy() == 10.0)

    ds3 = 10.0 - ds1
    assert isinstance(ds3, mikeio.Dataset)
    assert np.all(ds3[0].to_numpy() == 9.9)
    assert np.all(ds3[1].to_numpy() == 9.8)


def test_mul_scalar(ds1: Dataset) -> None:
    ds2 = ds1 * 2.0
    assert np.all(ds2[0].to_numpy() * 0.5 == ds1[0].to_numpy())

    ds3 = 2.0 * ds1
    assert np.all(ds3[0].to_numpy() == ds2[0].to_numpy())
    assert np.all(ds3[1].to_numpy() == ds2[1].to_numpy())


def test_add_dataset(ds1: Dataset, ds2: Dataset) -> None:
    ds3 = ds1 + ds2
    assert np.all(ds3[0].to_numpy() == 1.1)
    assert np.all(ds3[1].to_numpy() == 2.2)

    ds4 = ds2 + ds1
    assert np.all(ds3[0].to_numpy() == ds4[0].to_numpy())
    assert np.all(ds3[1].to_numpy() == ds4[1].to_numpy())

    ds2b = ds2.copy()
    ds2b[0].item = ItemInfo(EUMType.Wind_Velocity)
    # item type does not match, but we don't care about the item type, item is defined by the first dataset
    ds3 = ds2b + ds1
    assert ds3.items[0].type == EUMType.Wind_Velocity
    assert ds3.items[0].name == ds2b.items[0].name

    ds4 = ds1 + ds2b
    assert ds4.items[0].type == EUMType.Undefined
    assert ds4.items[0].name == ds1.items[0].name
    ds2c = ds2.copy()
    tt = ds2c.time.to_numpy()
    tt[-1] = tt[-1] + np.timedelta64(1, "s")
    ds2c.time = pd.DatetimeIndex(tt)
    with pytest.raises(ValueError):
        # time does not match
        ds1 + ds2c


def test_sub_dataset(ds1: Dataset, ds2: Dataset) -> None:
    ds3 = ds2 - ds1
    assert np.all(ds3[0].to_numpy() == 0.9)
    assert np.all(ds3[1].to_numpy() == 1.8)


def test_multiply_dataset(ds1: Dataset, ds2: Dataset) -> None:
    dsa = mikeio.Dataset(
        {
            "Foo": mikeio.DataArray(
                [1, 2, 3], item=mikeio.ItemInfo("Foo", EUMType.Water_Level)
            )
        }
    )
    dsb = mikeio.Dataset({"Foo": mikeio.DataArray([4, 5, 6])})
    dsr = dsa * dsb
    assert np.all(dsr["Foo"].to_numpy() == np.array([4, 10, 18]))
    assert dsr["Foo"].type == EUMType.Water_Level


def test_multiply_number_of_items_datasets_must_match() -> None:
    dsa = mikeio.Dataset(
        {"Foo": mikeio.DataArray([1, 2, 3]), "Bar": mikeio.DataArray([1, 2, 3])}
    )
    dsb = mikeio.Dataset({"Bar": mikeio.DataArray([4, 5, 6])})
    with pytest.raises(ValueError, match="Number of items"):
        dsa * dsb


def test_divide_dataset(ds1: Dataset, ds2: Dataset) -> None:
    ds_nom = mikeio.Dataset({"Foo": mikeio.DataArray([1, 2, 3])})
    ds_denom = mikeio.Dataset({"Foo": mikeio.DataArray([4, 5, 6])})
    ds3 = ds_nom / ds_denom
    assert np.all(ds3[0].to_numpy() == np.array([0.25, 0.4, 0.5]))


def test_divide_number_of_items_datasets_must_match() -> None:
    dsa = mikeio.Dataset(
        {"Foo": mikeio.DataArray([1, 2, 3]), "Bar": mikeio.DataArray([1, 2, 3])}
    )
    dsb = mikeio.Dataset({"Bar": mikeio.DataArray([4, 5, 6])})
    with pytest.raises(ValueError, match="Number of items"):
        dsa / dsb


def test_non_equidistant() -> None:
    nt = 4
    d = np.random.uniform(size=nt)

    ds = mikeio.Dataset.from_numpy(
        data=[d],
        time=[
            datetime(2000, 1, 1),
            datetime(2001, 1, 1),
            datetime(2002, 1, 1),
            datetime(2003, 1, 1),
        ],
    )

    assert not ds.is_equidistant


def test_concat_dataarray_by_time() -> None:
    da1 = mikeio.read("tests/testdata/tide1.dfs1")[0]
    da2 = mikeio.read("tests/testdata/tide2.dfs1")[0]
    da3 = mikeio.DataArray.concat([da1, da2])

    assert da3.start_time == da1.start_time
    assert da3.start_time < da2.start_time
    assert da3.end_time == da2.end_time
    assert da3.end_time > da1.end_time
    assert da3.n_timesteps == 145
    assert da3.is_equidistant


def test_concat_dataarray_keep_first() -> None:
    da1 = mikeio.DataArray(
        data=np.array([1.0, 2.0, 3.0]), time=pd.date_range("2000-01-01", periods=3)
    )
    # another dataarray with partly overlapping time
    da2 = mikeio.DataArray(
        data=np.array([4.0, 5.0]), time=pd.date_range("2000-01-02", periods=2)
    )

    da3 = mikeio.DataArray.concat([da1, da2], keep="first")

    assert da3.n_timesteps == 3
    assert da3.to_numpy()[2] == 3.0


def test_concat_by_time() -> None:
    ds1 = mikeio.read("tests/testdata/tide1.dfs1")
    ds2 = mikeio.read("tests/testdata/tide2.dfs1") + 0.5  # add offset
    ds3 = mikeio.Dataset.concat([ds1, ds2])

    assert isinstance(ds3, mikeio.Dataset)
    assert len(ds1) == len(ds2) == len(ds3)
    assert ds3.start_time == ds1.start_time
    assert ds3.start_time < ds2.start_time
    assert ds3.end_time == ds2.end_time
    assert ds3.end_time > ds1.end_time
    assert ds3.n_timesteps == 145
    assert ds3.is_equidistant


def test_concat_by_time_ndim1() -> None:
    ds1 = mikeio.read("tests/testdata/tide1.dfs1").isel(x=0)
    ds2 = mikeio.read("tests/testdata/tide2.dfs1").isel(x=0)
    ds3 = mikeio.Dataset.concat([ds1, ds2])

    assert isinstance(ds3, mikeio.Dataset)
    assert len(ds1) == len(ds2) == len(ds3)
    assert ds3.start_time == ds1.start_time
    assert ds3.start_time < ds2.start_time
    assert ds3.end_time == ds2.end_time
    assert ds3.end_time > ds1.end_time
    assert ds3.n_timesteps == 145
    assert ds3.is_equidistant


def test_concat_by_time_inconsistent_shape_not_possible() -> None:
    ds1 = mikeio.read("tests/testdata/tide1.dfs1").isel(x=[0, 1])
    ds2 = mikeio.read("tests/testdata/tide2.dfs1").isel(x=[0, 1, 2])
    with pytest.raises(ValueError, match="Shape"):
        mikeio.Dataset.concat([ds1, ds2])


def test_concat_dataset_different_items_not_possible() -> None:
    ds1 = mikeio.read("tests/testdata/HD2D.dfsu")
    ds2 = mikeio.read("tests/testdata/HD2D.dfsu", items=[1, 2])
    with pytest.raises(ValueError, match="items"):
        mikeio.Dataset.concat([ds1, ds2])


# TODO: implement this
def test_concat_by_time_no_time() -> None:
    ds1 = mikeio.read("tests/testdata/tide1.dfs1", time=0)
    ds2 = mikeio.read("tests/testdata/tide2.dfs1", time=1)
    ds3 = mikeio.Dataset.concat([ds1, ds2])

    assert ds3.n_timesteps == 2


def test_concat_by_time_2() -> None:
    ds1 = mikeio.read("tests/testdata/tide1.dfs1", time=range(0, 12))
    ds2 = mikeio.read("tests/testdata/tide2.dfs1")
    ds3 = mikeio.Dataset.concat([ds1, ds2])

    assert ds3.n_timesteps == 109
    assert not ds3.is_equidistant

    # create concatd datasets in 8 chunks of 6 hours
    dsall = []
    for j in range(8):
        dsall.append(
            mikeio.read(
                "tests/testdata/tide1.dfs1", time=range(j * 12, 1 + (j + 1) * 12)
            )
        )
    ds4 = mikeio.Dataset.concat(dsall)
    assert len(dsall) == 8
    assert ds4.n_timesteps == 97
    assert ds4.is_equidistant


def test_renamed_dataset_has_updated_attributes(ds1: mikeio.Dataset) -> None:
    assert hasattr(ds1, "Foo")
    assert isinstance(ds1.Foo, mikeio.DataArray)
    ds2 = ds1.rename(dict(Foo="Baz"))
    assert not hasattr(ds2, "Foo")
    assert hasattr(ds2, "Baz")
    assert isinstance(ds2.Baz, mikeio.DataArray)

    # inplace version
    ds1.rename(dict(Foo="Baz"), inplace=True)
    assert not hasattr(ds1, "Foo")
    assert hasattr(ds1, "Baz")
    assert isinstance(ds1.Baz, mikeio.DataArray)


def test_merge_by_item() -> None:
    ds1 = mikeio.read("tests/testdata/tide1.dfs1")
    ds2 = mikeio.read("tests/testdata/tide1.dfs1")
    old_name = ds2[0].name
    new_name = old_name + " v2"
    # ds2[0].name = ds2[0].name + " v2"
    ds2.rename({old_name: new_name}, inplace=True)
    ds3 = mikeio.Dataset.merge([ds1, ds2])

    assert isinstance(ds3, mikeio.Dataset)
    assert ds3.n_items == 2
    assert ds3[1].name == ds1[0].name + " v2"


def test_merge_must_have_same_time() -> None:
    ds1 = mikeio.Dataset(
        {
            "Foo": mikeio.DataArray(
                data=np.random.rand(10), time=pd.date_range("2000-01-01", periods=10)
            )
        }
    )
    ds2 = mikeio.Dataset(
        {
            "Bar": mikeio.DataArray(
                data=np.random.rand(10), time=pd.date_range("2100-01-01", periods=10)
            )
        }
    )
    with pytest.raises(ValueError, match="timesteps"):
        mikeio.Dataset.merge([ds1, ds2])


def test_merge_by_item_dfsu_3d() -> None:
    ds1 = mikeio.read("tests/testdata/oresund_sigma_z.dfsu", items=[0])
    assert ds1.n_items == 1
    ds2 = mikeio.read("tests/testdata/oresund_sigma_z.dfsu", items=[1])
    assert ds2.n_items == 1

    ds3 = mikeio.Dataset.merge([ds1, ds2])

    assert isinstance(ds3, mikeio.Dataset)
    itemnames = [x.name for x in ds3.items]
    assert "Salinity" in itemnames
    assert "Temperature" in itemnames
    assert ds3.n_items == 2


def test_to_numpy(ds2: Dataset) -> None:
    X = ds2.to_numpy()

    assert X.shape == (ds2.n_items,) + ds2.shape
    assert isinstance(X, np.ndarray)


def test_concat() -> None:
    filename = "tests/testdata/HD2D.dfsu"
    dss1 = mikeio.read(filename, time=[0, 1])
    dss2 = mikeio.read(filename, time=[2, 3])
    dss3 = mikeio.Dataset.concat([dss1, dss2])

    assert dss1.n_items == dss2.n_items == dss3.n_items
    assert dss3.n_timesteps == (dss1.n_timesteps + dss2.n_timesteps)
    assert dss3.start_time == dss1.start_time
    assert dss3.end_time == dss2.end_time
    assert isinstance(dss1.geometry, mikeio.spatial.GeometryFM2D)
    assert isinstance(dss3.geometry, mikeio.spatial.GeometryFM2D)
    assert dss3.geometry.n_elements == dss1.geometry.n_elements


def test_concat_dfsu3d() -> None:
    filename = "tests/testdata/basin_3d.dfsu"
    ds = mikeio.read(filename)
    ds1 = mikeio.read(filename, time=[0, 1])
    ds2 = mikeio.read(filename, time=[1, 2])
    ds3 = mikeio.Dataset.concat([ds1, ds2])

    assert ds1.n_items == ds2.n_items == ds3.n_items
    assert ds3.start_time == ds.start_time
    assert ds3.end_time == ds.end_time
    assert isinstance(ds1.geometry, mikeio.spatial.GeometryFM3D)
    assert isinstance(ds3.geometry, mikeio.spatial.GeometryFM3D)
    assert ds3.geometry.n_elements == ds1.geometry.n_elements
    assert ds3._zn.shape == ds._zn.shape  # type: ignore
    assert np.all(ds3._zn == ds._zn)


def test_concat_dfsu3d_single_timesteps() -> None:
    filename = "tests/testdata/basin_3d.dfsu"
    mikeio.read(filename)
    ds1 = mikeio.read(filename, time=0)
    ds2 = mikeio.read(filename, time=2)
    ds3 = mikeio.Dataset.concat([ds1, ds2])

    assert ds1.n_items == ds2.n_items == ds3.n_items
    assert ds3.start_time == ds1.start_time
    assert ds3.end_time == ds2.end_time


def test_concat_dfs2_single_timesteps() -> None:
    filename = "tests/testdata/single_row.dfs2"
    mikeio.read(filename)
    ds1 = mikeio.read(filename, time=0)
    ds2 = mikeio.read(filename, time=2)
    ds3 = mikeio.Dataset.concat([ds1, ds2])

    assert ds1.n_items == ds2.n_items == ds3.n_items
    assert ds3.start_time == ds1.start_time
    assert ds3.end_time == ds2.end_time
    assert ds3.n_timesteps == 2


def test_merge_same_name_error() -> None:
    filename = "tests/testdata/HD2D.dfsu"
    ds1 = mikeio.read(filename, items=0)
    ds2 = mikeio.read(filename, items=0)

    assert ds1.items[0].name == ds2.items[0].name

    with pytest.raises(ValueError):
        mikeio.Dataset.merge([ds1, ds2])


def test_incompatible_data_not_allowed() -> None:
    da1 = mikeio.read("tests/testdata/HD2D.dfsu")[0]
    da2 = mikeio.read("tests/testdata/oresundHD_run1.dfsu")[1]

    with pytest.raises(ValueError) as excinfo:
        mikeio.Dataset([da1, da2])

    assert "shape" in str(excinfo.value).lower()


def test_xzy_selection() -> None:
    # select in space via x,y,z coordinates test
    filename = "tests/testdata/oresund_sigma_z.dfsu"
    ds = mikeio.read(filename)

    with pytest.raises(OutsideModelDomainError):
        ds.sel(x=340000, y=15.75, z=0)


def test_layer_selection() -> None:
    # select layer test
    filename = "tests/testdata/oresund_sigma_z.dfsu"
    ds = mikeio.read(filename)

    dss_layer = ds.sel(layers=0)
    # should not be layered after selection
    assert type(dss_layer.geometry) is mikeio.spatial.GeometryFM2D


def test_time_selection() -> None:
    # select time test
    nt = 100
    data = []
    d = np.random.rand(nt)
    data.append(d)
    time = pd.date_range("2000-1-2", freq="h", periods=nt)
    items = [ItemInfo("Foo")]
    ds = mikeio.Dataset.from_numpy(data=data, time=time, items=items)

    # check for string input
    dss_t = ds.sel(time="2000-01-05")
    # and index based
    dss_tix = ds.sel(time=80)

    assert dss_t.shape == (24,)
    assert len(dss_tix) == 1


def test_create_dataset_with_many_items() -> None:
    n_items = 800
    nt = 2
    time = pd.date_range("2000", freq="h", periods=nt)

    das = []

    for i in range(n_items):
        x = np.random.random(nt)
        da = mikeio.DataArray(data=x, time=time, item=mikeio.ItemInfo(f"Item {i + 1}"))
        das.append(da)

    ds = mikeio.Dataset(das)

    assert ds.n_items == n_items


def test_create_array_with_defaults_from_dataset() -> None:
    filename = "tests/testdata/oresund_sigma_z.dfsu"
    ds: mikeio.Dataset = mikeio.read(filename)

    values = np.zeros(ds["Temperature"].shape)

    da = ds.create_data_array(values)

    assert isinstance(da, mikeio.DataArray)
    assert da.geometry == ds.geometry
    assert all(da.time == ds.time)
    assert da.item.type == mikeio.EUMType.Undefined

    da_name = ds.create_data_array(values, name="Foo")

    assert isinstance(da, mikeio.DataArray)
    assert da_name.geometry == ds.geometry
    assert da_name.item.type == mikeio.EUMType.Undefined
    assert da_name.name == "Foo"

    da_eum = ds.create_data_array(
        values, item=mikeio.ItemInfo("TS", mikeio.EUMType.Temperature)
    )

    assert isinstance(da_eum, mikeio.DataArray)
    assert da_eum.geometry == ds.geometry
    assert da_eum.item.type == mikeio.EUMType.Temperature


def test_dataset_plot(ds1: Dataset) -> None:
    ax = ds1.isel(x=0).plot()
    assert len(ax.lines) == 2
    ds2 = ds1 + 0.01
    ax = ds2.isel(x=-1).plot(ax=ax)
    assert len(ax.lines) == 4


def test_interp_na() -> None:
    time = pd.date_range("2000", periods=5, freq="D")
    da = mikeio.DataArray(
        data=np.array([np.nan, 1.0, np.nan, np.nan, 4.0]),
        time=time,
        item=ItemInfo(name="Foo"),
    )
    da2 = mikeio.DataArray(
        data=np.array([np.nan, np.nan, np.nan, 4.0, 4.0]),
        time=time,
        item=ItemInfo(name="Bar"),
    )

    ds = mikeio.Dataset([da, da2])

    dsi = ds.interp_na()
    assert np.isnan(dsi[0].to_numpy()[0])
    assert dsi.Foo.to_numpy()[2] == pytest.approx(2.0)  # type: ignore
    assert np.isnan(dsi.Foo.to_numpy()[0])  # type: ignore

    dsi = ds.interp_na(fill_value="extrapolate")
    assert dsi.Foo.to_numpy()[0] == pytest.approx(0.0)  # type: ignore
    assert dsi.Bar.to_numpy()[2] == pytest.approx(4.0)  # type: ignore


def test_plot_scatter() -> None:
    ds = mikeio.read("tests/testdata/oresund_sigma_z.dfsu", time=0)
    ds.plot.scatter(x="Salinity", y="Temperature", title="S-vs-T")


def test_select_single_timestep_preserves_dt() -> None:
    ds = mikeio.read("tests/testdata/tide1.dfs1")
    assert ds.timestep == pytest.approx(1800.0)
    ds2 = ds.isel(time=-1)
    assert ds2.timestep == pytest.approx(1800.0)
    assert ds2[0].timestep == pytest.approx(1800.0)


def test_select_multiple_spaced_timesteps_uses_proper_dt(tmp_path: Path) -> None:
    ds = mikeio.read("tests/testdata/tide1.dfs1")
    assert ds.timestep == pytest.approx(1800.0)
    ds2 = ds.isel(time=[0, 2, 4])
    assert ds2.timestep == pytest.approx(3600.0)


def test_read_write_single_timestep_preserves_dt(tmp_path: Path) -> None:
    fn = "tests/testdata/oresund_sigma_z.dfsu"
    dfs = mikeio.Dfsu3D(fn)
    assert dfs.timestep == pytest.approx(10800.0)

    ds = dfs.read(time=[0])
    assert ds.timestep == pytest.approx(dfs.timestep)

    outfn = tmp_path / "single.dfsu"
    ds.to_dfs(outfn)

    dfs2 = mikeio.Dfsu3D(outfn)
    assert dfs2.timestep == pytest.approx(10800.0)


def test_fillna() -> None:
    ds = mikeio.Dataset(
        {
            "foo": mikeio.DataArray(np.array([np.nan, 1.0])),
            "bar": mikeio.DataArray(np.array([2.0, np.nan])),
            "baz": mikeio.DataArray(np.array([2.0, 3.0])),
        }
    )
    assert np.isnan(ds["foo"].to_numpy()[0])
    assert np.isnan(ds["bar"].to_numpy()[-1])

    ds_filled = ds.fillna()

    assert ds_filled["foo"].to_numpy()[0] == pytest.approx(0.0)
    assert ds_filled["bar"].to_numpy()[-1] == pytest.approx(0.0)

    # original dataset is not modified
    assert np.isnan(ds["foo"].to_numpy()[0])


def test_safe_name() -> None:
    from mikeio.dataset._dataset import _to_safe_name

    good_name = "MSLP"

    assert _to_safe_name(good_name) == good_name

    bad_name = "MSLP., 1:st level\n 2nd chain"
    safe_name = "MSLP_1_st_level_2nd_chain"
    assert _to_safe_name(bad_name) == safe_name
