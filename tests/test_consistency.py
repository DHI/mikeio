import mikeio


def test_read_dfsu():
    ds = mikeio.read(
        "tests/testdata/consistency/oresundHD.dfsu",
        items=[0, 1],
        time_steps=slice("2018", "2018-03-10"),
    )

    assert ds.n_items == 2
    assert ds.n_timesteps == 4


def test_read_dfs0():
    ds = mikeio.read(
        "tests/testdata/consistency/oresundHD.dfs0",
        items=[0, 1],
        time_steps=slice("2018", "2018-03-10"),
    )

    assert ds.n_items == 2
    assert ds.n_timesteps == 4


def test_read_dfs1():
    ds = mikeio.read(
        "tests/testdata/consistency/oresundHD.dfs1",
        items=[0, 1],
        time_steps=slice("2018", "2018-03-10"),
    )

    assert ds.n_items == 2
    assert ds.n_timesteps == 4


def test_read_dfs2():
    ds = mikeio.read(
        "tests/testdata/consistency/oresundHD.dfs2",
        items=[0, 1],
        time_steps=slice("2018", "2018-03-10"),
    )

    assert ds.n_items == 2
    assert ds.n_timesteps == 4
