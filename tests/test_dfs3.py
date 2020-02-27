from mikeio.dfs3 import Dfs3


# TODO Rewrite dfs3 to return data in (t, z, y, x) order
def test_read_dfs3():
    dfs = Dfs3()
    (data, t, names) = dfs.read("tests/testdata/Grid1.dfs3")

    assert len(data) == 2
    assert data[0].shape == (
        10,
        10,
        10,
        30,
    )  # Note still inconsistent with dfs1, dfs2, dfsu

