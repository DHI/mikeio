import mikeio


def test_pfs_with_multipolygon() -> None:
    pfs = mikeio.read_pfs("tests/testdata/pfs/multipolygon.pfs")
    assert pfs.n_targets == 1
