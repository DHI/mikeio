import os
import numpy as np

import pytest

from mikeio import Dfs0


def test_simple_write_big_file(tmpdir):

    filename = os.path.join(tmpdir.dirname, "big.dfs0")
    
    data = []

    nt = 10_000_000
    d = np.random.random([nt])
    data.append(d)

    dfs = Dfs0()

    dfs.write(filename=filename, data=data)

    assert os.path.exists(filename)

def test_simple_write_read_big_file(tmpdir):

    filename = os.path.join(tmpdir.dirname, "big.dfs0")
    
    data = []

    nt = 10_000_000
    d = np.random.random([nt])
    data.append(d)

    dfs = Dfs0()

    dfs.write(filename=filename, data=data)

    assert os.path.exists(filename)

    dfs = Dfs0(filename=filename)
    ds = dfs.read()

    assert len(ds.time) == nt