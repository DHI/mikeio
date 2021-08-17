from datetime import datetime
from mikecore.eum import eumUnit
from mikeio.eum import EUMType, ItemInfo
import numpy as np

from mikeio import Dfs2

dfs = Dfs2()

# boogus data, in reality you would get this from some other file, like another dfs2 or netcdf...
data = [
    np.full(shape=(1, 200, 800), fill_value=0.0),
    np.random.random(size=[1, 200, 800]),
]

nt = 1000

with dfs.write(
    "long_big_file.dfs2",
    data,
    start_time=datetime(2021, 1, 1),
    dt=3600,
    items=[
        ItemInfo(EUMType.Water_Level),
        ItemInfo(EUMType.Temperature),
    ],
    coordinate=["UTM-33", 12.0, 55.0, 0.0],
    dx=100,
    dy=200,
    keep_open=True,
) as f:

    for i in range(1, nt):
        # boogus data, in reality you would get this from some other file, like another dfs2 or netcdf...
        data = [
            np.full(shape=(1, 200, 800), fill_value=float(i)),
            np.random.random(size=[1, 200, 800]),
        ]
        f.append(data)
