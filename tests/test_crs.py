from mikeio.spatial import CRS

def test_utm_32():

    crs = CRS("UTM-32") # Convenient short-hand for "PROJCS["WGS 84 / UTM zone 32N"..."

    assert crs.is_projected
    assert not crs.is_geographic