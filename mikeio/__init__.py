import sys
import os
from platform import architecture

# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# Generic release markers:
#   X.Y
#   X.Y.Z   # For bugfix releases
#
# Admissible pre-release markers:
#   X.YaN   # Alpha release
#   X.YbN   # Beta release
#   X.YrcN  # Release Candidate
#   X.Y     # Final release
#
# Dev branch marker is: 'X.Y.dev' or 'X.Y.devN' where N is an integer.
# 'X.Y.dev0' is the canonical version of 'X.Y.dev'
#

__version__ = "1.0.0"
__dfs_version__: int = 100


if "64" not in architecture()[0]:
    raise Exception("This library has not been tested for a 32 bit system.")

from .dfs0 import Dfs0
from .dfs1 import Dfs1
from .dfs2 import Dfs2
from .dfs3 import Dfs3
from .dfsu_factory import Dfsu
from .dfsu import Mesh
from .pfs import Pfs
from .xyz import read_xyz
from .dataset import Dataset, DataArray
from .spatial.grid_geometry import Grid1D, Grid2D, Grid3D
from .eum import ItemInfo, EUMType, EUMUnit


def read(
    filename, *, items=None, time_steps=None, time=None, keepdims=False, **kwargs
) -> Dataset:
    """Read all or a subset of the data from a dfs file

    All dfs files can be subsetted with the *items* and *time* arguments. But
    the following file types also have the shown additional arguments:

    * Dfs2: area
    * Dfs3: layers
    * Dfsu-2d: (x,y), elements, area
    * Dfsu-layered: (xy,z), elements, area, layers

    Parameters
    ----------
    filename
        full path and file name to the dfs file.
    items: int, str, list[int] or list[str], optional
        Read only selected items, by number (0-based), or by name,
        by default None (=all)
    time: int, str, datetime, pd.TimeStamp, sequence, slice or pd.DatetimeIndex, optional
        Read only selected time steps, by default None (=all)
    keepdims: bool, optional
        When reading a single time step only, should the time-dimension be kept
        in the returned Dataset? by default: False
    x, y, z: float, optional
        Dfsu: Read only data for elements containing the (x,y)
        or (x,y,z) points(s), by default None
    area: (float, float, float, float), optional
        Dfs2/Dfsu: read only data within an area given by a bounding
        box of coordinates (left, lower, right, upper), by default None (=all)
    layers: int, str or sequence, optional
        Dfs3/Dfsu-layered: read only data from specific layers,
        by default None (=all layers)

    Returns
    -------
    Dataset
        A Dataset with specification according to the file type

    See also
    --------
    mikeio.open - open a Dfs file and only read the header

    Examples
    --------
    >>> ds = mikeio.read("")
    """

    _, ext = os.path.splitext(filename)

    if "dfs" not in ext:
        raise ValueError("mikeio.read is only supported for Dfs files")

    dfs = open(filename)

    return dfs.read(
        items=items, time_steps=time_steps, time=time, keepdims=keepdims, **kwargs
    )


def open(filename: str, **kwargs):
    """Open a dfs/mesh file

    Examples
    --------
    >>> dfs = mikeio.open("wl.dfs1")
    >>> dfs = mikeio.open("HD2D.dfsu")
    >>> dfs = mikeio.open("HD2D.dfsu", dtype=np.float64)
    """
    _, ext = os.path.splitext(filename)

    if ext == ".dfs0":
        return Dfs0(filename, **kwargs)

    elif ext == ".dfs1":
        return Dfs1(filename, **kwargs)

    elif ext == ".dfs2":
        return Dfs2(filename, **kwargs)

    elif ext == ".dfs3":
        return Dfs3(filename, **kwargs)

    elif ext == ".dfsu":
        return Dfsu(filename, **kwargs)

    elif ext == ".mesh":
        return Mesh(filename, **kwargs)

    else:
        raise Exception(f"{ext} is an unsupported extension")
