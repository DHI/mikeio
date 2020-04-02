from mikeio.dfs3 import Dfs3


def test_read_dfs3():
    dfs = Dfs3()
    ds = dfs.read("tests/testdata/Grid1.dfs3")

    assert len(ds.data) == 2
    assert len(ds.time) == 30
    assert ds.data[0].shape == (30, 10, 10, 10)  # t  # z  # y  # x
