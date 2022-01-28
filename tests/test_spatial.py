import numpy as np
from mikeio.spatial import dist_in_meters, min_horizontal_dist_meters


def test_dist_in_meters():

    np.random.seed = 42
    n = 1000
    lon = np.random.uniform(low=-195, high=400, size=n)
    lat = np.random.uniform(low=-89, high=89, size=n)
    coords = np.vstack([lon, lat]).T
    poi = [0.0, 0.0]
    dist = dist_in_meters(coords, poi, is_geo=True)
    print(dist.max)
    assert dist.shape == (n,)
    assert dist.max() < 20015100

    dist = dist_in_meters(coords, poi, is_geo=False)
    assert dist.shape == (n,)


def test_min_horizontal_dist_meters():
    n = 11
    lon = np.linspace(0, 10, n)
    lat = np.linspace(50, 52, n)
    coords = np.vstack([lon, lat]).T

    lon = np.linspace(0, 10, 3)
    lat = np.linspace(52, 54, 3)
    targets = np.vstack([lon, lat]).T

    min_d = min_horizontal_dist_meters(coords, targets, is_geo=True)
    assert min_d.shape == (n,)
    assert min_d[0] == 222389.85328911748
