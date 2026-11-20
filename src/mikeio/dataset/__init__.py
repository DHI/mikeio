from typing import TYPE_CHECKING

from ._dataarray import DataArray
from ._dataset import Dataset, from_pandas, from_polars

if TYPE_CHECKING:
    from ._data_plot import (
        DataArrayPlotter,
        DataArrayPlotterGrid1D,
        DataArrayPlotterGrid2D,
        DataArrayPlotterFM,
        DataArrayPlotterFMVerticalColumn,
        DataArrayPlotterFMVerticalProfile,
        DataArrayPlotterPointSpectrum,
        DataArrayPlotterLineSpectrum,
        DataArrayPlotterAreaSpectrum,
        DatasetPlotter,
    )

__all__ = [
    "DataArray",
    "Dataset",
    "from_pandas",
    "from_polars",
    "DataArrayPlotter",
    "DataArrayPlotterGrid1D",
    "DataArrayPlotterGrid2D",
    "DataArrayPlotterFM",
    "DataArrayPlotterFMVerticalColumn",
    "DataArrayPlotterFMVerticalProfile",
    "DataArrayPlotterPointSpectrum",
    "DataArrayPlotterLineSpectrum",
    "DataArrayPlotterAreaSpectrum",
    "DatasetPlotter",
]

# Plotter classes import matplotlib, which is heavy; load them on first use.
_PLOTTER_NAMES = frozenset(__all__[4:])


def __getattr__(name: str) -> type:
    if name in _PLOTTER_NAMES:
        from . import _data_plot

        return getattr(_data_plot, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
