import os
import mikeio
from mikeio.aggregator import dfs2todfs1, dfstodfs0


def test_dfs2_to_dfs1_y_direction(tmpdir):

    dfs2file = "tests/testdata/eq.dfs2"

    ds2 = mikeio.read(dfs2file)

    assert ds2[0].values.shape == (25, 10, 20)  # t, y, x

    dfs1file1 = os.path.join(tmpdir.dirname, "eq_ax_y.dfs1")
    dfs2todfs1(dfs2file, dfs1file1)  # default aggregation is over y axis

    ds1 = mikeio.read(dfs1file1)

    assert ds1[0].values.shape == (25, 20)


def test_dfs2_to_dfs1_x_direction(tmpdir):

    dfs2file = "tests/testdata/eq.dfs2"

    ds2 = mikeio.read(dfs2file)

    assert ds2[0].values.shape == (25, 10, 20)

    dfs1file1 = os.path.join(tmpdir.dirname, "eq_ax_x.dfs1")
    dfs2todfs1(dfs2file, dfs1file1, axis=2)

    ds1 = mikeio.read(dfs1file1)

    assert ds1[0].values.shape == (25, 10)


def test_dfsu_to_dfs0(tmpdir):

    dfsufile = "tests/testdata/HD2D.dfsu"

    ds_in = mikeio.read(dfsufile)

    assert ds_in[0].values.shape == (9, 884)

    dfs0file1 = os.path.join(tmpdir.dirname, "HD2D_mean.dfs0")
    dfstodfs0(dfsufile, dfs0file1)

    ds0 = mikeio.read(dfs0file1)

    assert ds0[0].values.shape == (9,)
