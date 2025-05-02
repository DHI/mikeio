from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from mikeio.spatial import GeometryFM2D, GeometryFM3D
from ._dataarray import DataArray
from ._dataset import Dataset
from ..eum import EUMType, EUMUnit, ItemInfo

class GeometryFM2DDataArray(DataArray):

    geometry: GeometryFM2D

    def __init__(
        self,
        data: np.ndarray,
        *,
        time: pd.DatetimeIndex | str | None = None,
        name: str | None = None,
        type: EUMType | None = None,
        unit: EUMUnit | None = None,
        item: ItemInfo | None = None,
        geometry: GeometryFM2D | None = None,
        zn: np.ndarray | None = None,
        dims: Sequence[str] | None = None,
        dt: float = 1.0,
    ) -> None:
        super().__init__(data=data,time=time, name=name, type=type, unit=unit, item=item,geometry=geometry, zn=zn, dims=dims, dt=dt)

class GeometryFM2DDataset(Dataset[GeometryFM2DDataArray, GeometryFM2D]):
    def __init__(self, data: Mapping[str, GeometryFM2DDataArray] | Sequence[GeometryFM2DDataArray], validate:bool=False) -> None:
        super().__init__(data=data, validate=validate)

    @property
    def geometry(self) -> GeometryFM2D:
        """Geometry of each DataArray."""
        return self[0].geometry

class GeometryFM3DDataArray(DataArray):

    def __init__(
        self,
        data: np.ndarray,
        *,
        time: pd.DatetimeIndex | str | None = None,
        name: str | None = None,
        type: EUMType | None = None,
        unit: EUMUnit | None = None,
        item: ItemInfo | None = None,
        geometry: GeometryFM3D | None = None,
        zn: np.ndarray | None = None,
        dims: Sequence[str] | None = None,
        dt: float = 1.0,
    ) -> None:
        super().__init__(data=data,time=time, name=name, type=type, unit=unit, item=item,geometry=geometry, zn=zn, dims=dims, dt=dt)

class GeometryFM3DDataset(Dataset[GeometryFM2DDataArray]):
    def __init__(self, data: Mapping[str, GeometryFM3DDataArray] | Sequence[GeometryFM3DDataArray], validate:bool=False) -> None:
        super().__init__(data=data, validate=validate)

    @property
    def geometry(self) -> GeometryFM3D:
        """Geometry of each DataArray."""
        return self[0].geometry