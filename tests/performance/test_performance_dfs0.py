from pathlib import Path
import numpy as np
import pandas as pd
import mikeio


def test_write_read_long_dfs0(tmp_path: Path) -> None:
    filename = tmp_path / "big.dfs0"

    nt = 10_000_000
    data = np.random.random([nt])
    da = mikeio.DataArray(
        data=data, time=pd.date_range(start="2001-01-01", freq="s", periods=nt)
    )
    da.to_dfs(filename)

    assert filename.exists()

    ds = mikeio.read(filename)

    assert len(ds.time) == nt


def test_write_read_many_items_dataset_pandas(tmp_path: Path) -> None:
    filename = tmp_path / "many_items.dfs0"

    n_items = 10_000
    nt = 200
    time = pd.date_range("2000", freq="s", periods=nt)
    das = [
        mikeio.DataArray(
            data=np.random.random([nt]), time=time, item=mikeio.ItemInfo(f"Item {i+1}")
        )
        for i in range(n_items)
    ]
    ds = mikeio.Dataset(das)
    ds.to_dfs(filename)

    assert filename.exists()

    # read to dataset
    ds = mikeio.read(filename)

    assert len(ds.time) == nt
    assert ds.n_items == n_items

    # skip dataset, read directly to dataframe
    dfs = mikeio.Dfs0(filename=filename)
    df = dfs.to_dataframe()

    assert len(df) == nt
    assert len(df.columns) == n_items
