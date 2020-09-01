from mikeio.spatial import dist_in_meters
import numpy as np


def test_dist_in_meters():

    np.random.seed = 42
    n = 100
    lat = np.random.uniform(low=-89, high=89, size=n)
    lon = np.random.uniform(low=-179, high=179, size=n)
    coords = np.vstack([lat, lon]).T
    poi = [0.0, 0.0]
    dist = dist_in_meters(coords, poi, is_geo=True)
    print(dist.max)
    assert dist.shape == (n,)
    assert dist.max() < 20040000

