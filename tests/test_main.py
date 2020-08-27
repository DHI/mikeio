import pytest
import mikeio


def test_read_dfs0_generic_read():

    filename = r"tests/testdata/random.dfs0"

    res = mikeio.read(filename)
    data = res.data

    assert len(data) == 2


def test_read_dfs1_generic_read():

    filename = r"tests/testdata/random.dfs1"

    res = mikeio.read(filename)
    data = res.data[0]
    assert data.shape == (100, 3)  # time, x


def test_read_dfs2_generic_read():

    filename = r"tests/testdata/random.dfs2"

    res = mikeio.read(filename, ["testing water level"])
    data = res.data[0]
    assert data[0, 11, 0] == 0
    assert data.shape == (3, 100, 2)  # time, y, x


def test_read_dfsu_generic_read():

    filename = "tests/testdata/HD2D.dfsu"

    ds = mikeio.read(filename)

    assert len(ds) == 4


def test_read_dfsu_generic_read_single_item_number():

    filename = "tests/testdata/HD2D.dfsu"

    ds = mikeio.read(filename, [1])

    assert len(ds) == 1
    


def test_read_generic_read_unsupported_format():

    filename = "foo.txt"

    with pytest.raises(Exception):
        res = mikeio.read(filename)

