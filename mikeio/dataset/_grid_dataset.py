from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from mikeio.spatial._grid_geometry import Grid1D, Grid2D
from ._dataarray import DataArray
from ._dataset import Dataset
from ..eum import EUMType, EUMUnit, ItemInfo

class Grid1DDataArray(DataArray):

    def __init__(
        self,
        data: np.ndarray,
        *,
        time: pd.DatetimeIndex | str | None = None,
        name: str | None = None,
        type: EUMType | None = None,
        unit: EUMUnit | None = None,
        item: ItemInfo | None = None,
        geometry: Grid1D | None = None,
        zn: np.ndarray | None = None,
        dims: Sequence[str] | None = None,
        dt: float = 1.0,
    ) -> None:
        super().__init__(data=data,time=time, name=name, type=type, unit=unit, item=item,geometry=geometry, zn=zn, dims=dims, dt=dt)

class Grid1DDataset(Dataset[Grid1DDataArray, Grid1D]):
    def __init__(self, data: Mapping[str, Grid1DDataArray] | Sequence[Grid1DDataArray], validate:bool=False) -> None:
        super().__init__(data=data, validate=validate)
        

class Grid2DDataArray(DataArray):

    def __init__(
        self,
        data: np.ndarray,
        *,
        time: pd.DatetimeIndex | str | None = None,
        name: str | None = None,
        type: EUMType | None = None,
        unit: EUMUnit | None = None,
        item: ItemInfo | None = None,
        geometry: Grid2D | None = None,
        zn: np.ndarray | None = None,
        dims: Sequence[str] | None = None,
        dt: float = 1.0,
    ) -> None:
        super().__init__(data=data,time=time, name=name, type=type, unit=unit, item=item,geometry=geometry, zn=zn, dims=dims, dt=dt)

class Grid2DDataset(Dataset):
    def __init__(self, data: Mapping[str, Grid2DDataArray] | Sequence[Grid2DDataArray], validate:bool=False) -> None:
        super().__init__(data=data, validate=validate)