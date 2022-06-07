import os
import numpy as np
from datetime import datetime

from mikeio import Dfs0


def test_simple_write_big_file(tmpdir):

    filename = os.path.join(tmpdir.dirname, "big.dfs0")

    nt = 10_000_000
    data = [np.random.random([nt])]
    start_time = datetime(2001, 1, 1)
    Dfs0().write(filename=filename, data=data, start_time=start_time)

    assert os.path.exists(filename)


def test_simple_write_read_big_file(tmpdir):

    filename = os.path.join(tmpdir.dirname, "big.dfs0")

    nt = 10_000_000
    data = [np.random.random([nt])]
    start_time = datetime(2001, 1, 1)
    Dfs0().write(filename=filename, data=data, start_time=start_time)

    assert os.path.exists(filename)

    dfs = Dfs0(filename=filename)
    ds = dfs.read()

    assert len(ds.time) == nt


def test_write_many_items_dfs0(tmpdir):

    filename = os.path.join(tmpdir.dirname, "many_items.dfs0")

    n_items = 800
    nt = 1000
    data = [np.random.random([nt]) for _ in range(n_items)]
    start_time = datetime(2001, 1, 1)
    Dfs0().write(filename=filename, data=data, start_time=start_time)

    assert os.path.exists(filename)


def test_read_many_items_dfs0(tmpdir):

    filename = os.path.join(tmpdir.dirname, "many_items.dfs0")

    n_items = 800
    nt = 1000
    assert os.path.exists(filename)

    dfs = Dfs0(filename=filename)
    ds = dfs.read()

    assert len(ds.time) == nt
    assert ds.n_items == n_items


def test_write_read_many_items_dfs0_pandas(tmpdir):

    filename = os.path.join(tmpdir.dirname, "even_more_items.dfs0")

    n_items = 100_000  # pandas can read more...
    nt = 20
    data = [np.random.random([nt]) for _ in range(n_items)]
    start_time = datetime(2001, 1, 1)
    Dfs0().write(filename=filename, data=data, start_time=start_time)

    assert os.path.exists(filename)

    dfs = Dfs0(filename=filename)
    df = dfs.to_dataframe()

    assert len(df) == nt
    assert len(df.columns) == n_items
