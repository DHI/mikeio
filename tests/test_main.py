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

    res = mikeio.read(filename, item_names=["testing water level"])
    data = res.data[0]
    assert data[0, 11, 0] == 0
    assert data.shape == (3, 100, 2)  # time, y, x


def test_read_dfsu_generic_read():

    filename = "tests/testdata/HD2D.dfsu"

    (data, t, names) = mikeio.read(filename)

    assert len(data) == 4
    assert len(names) == 4


def test_read_dfsu_generic_read_single_item_number():

    filename = "tests/testdata/HD2D.dfsu"

    (data, t, names) = mikeio.read(filename, item_numbers=[1])

    assert len(data) == 1
    assert len(names) == 1


def test_read_generic_read_unsupported_format():

    filename = "foo.txt"

    with pytest.raises(Exception):
        res = mikeio.read(filename)

