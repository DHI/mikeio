import numpy as np
import mikeio


def test_find_index_inside_domain() -> None:
    g = mikeio.Grid2D(bbox=[-1, -1, 1, 5], dx=0.2)

    np.random.seed(0)

    # all points are inside the grid
    x = np.random.uniform(-1, 1, size=100_000)
    y = np.random.uniform(-1, 5, size=100_000)

    xy = np.column_stack([x, y])
    assert xy.shape == (100_000, 2)

    ii, jj = g.find_index(coords=xy)

    assert (ii[0], jj[0]) == g.find_index(x=x[0], y=y[0])
    assert (ii[-1], jj[-1]) == g.find_index(x=x[-1], y=y[-1])
