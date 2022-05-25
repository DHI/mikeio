import os
import pytest
import mikeio


##################################################
# these tests will not run if shapely is not installed
##################################################
pytest.importorskip("shapely")


def test_to_shapely():
    filename = os.path.join("tests", "testdata", "oresund_sigma_z.dfsu")
    dfs = mikeio.open(filename)
    shp = dfs.to_shapely()
    assert True
