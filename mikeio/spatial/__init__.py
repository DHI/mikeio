from ._geometry import GeometryPoint3D, GeometryPoint2D, GeometryUndefined
from ._FM_geometry import (
    GeometryFM2D,
    GeometryFMPointSpectrum,
    GeometryFMLineSpectrum,
    GeometryFMAreaSpectrum,
)

from ._FM_geometry_layered import (
    GeometryFMVerticalColumn,
    GeometryFMVerticalProfile,
    GeometryFM3D,
)

from ._grid_geometry import Grid1D, Grid2D, Grid3D


# from . import _FM_geometry as FM_geometry  # for backward compatibility

from ._FM_geometry import GeometryFM  # for backward compatibility
