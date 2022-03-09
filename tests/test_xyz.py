import mikeio


def test_read_xyz_tab():
    filename = "tests/testdata/positions.xyz"

    df = mikeio.read_xyz(filename)

    assert df.shape[1] == 4


def test_read_xyz_space():
    filename = "tests/testdata/winches.xyz"

    df = mikeio.read_xyz(filename)

    assert df.shape[1] == 3


def test_write_xyz(tmpdir):
    outfilename = tmpdir.join("foo.xyz")
    filename = "tests/testdata/positions.xyz"

    df1 = mikeio.read_xyz(filename)

    df1.to_xyz(outfilename)

    df2 = mikeio.read_xyz(outfilename)

    assert df1.shape == df2.shape
