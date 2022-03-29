import pytest
import mikeio


def test_read_dfs0_generic_read():

    filename = r"tests/testdata/random.dfs0"

    res = mikeio.read(filename)
    data = res.to_numpy()

    assert len(data) == 2


def test_read_dfs1_generic_read():

    filename = r"tests/testdata/random.dfs1"

    res = mikeio.read(filename)
    data = res[0].to_numpy()
    assert data.shape == (100, 3)  # time, x


def test_read_dfs2_generic_read():

    filename = "tests/testdata/gebco_sound.dfs2"

    ds = mikeio.read(filename)
    assert ds.geometry.nx == 216


def test_read_dfsu_generic_read():

    filename = "tests/testdata/HD2D.dfsu"

    ds = mikeio.read(filename)

    assert len(ds) == 4


# def test_read_dfsu_generic_read_single_item_number():

#   filename = "tests/testdata/HD2D.dfsu"

#    ds = mikeio.read(filename, [1])

#    assert len(ds) == 1


def test_read_generic_read_unsupported_format():

    filename = "foo.txt"

    with pytest.raises(Exception):
        res = mikeio.read(filename)
