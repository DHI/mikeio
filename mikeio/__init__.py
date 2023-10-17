from pathlib import Path
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

__version__ = "1.6.3"  # TODO use git hash instead for dev version?
# __version__ = "1.5.0"
__dfs_version__: int = 160


if "64" not in architecture()[0]:
    raise Exception("This library has not been tested for a 32 bit system.")

from .dataset import DataArray, Dataset
from .dfs import Dfs0, Dfs1, Dfs2, Dfs3
from .dfsu import Dfsu, Mesh
from .eum import EUMType, EUMUnit, ItemInfo
from .pfs import Pfs, PfsDocument, PfsSection, read_pfs

# Grid geometries are imported into the main module, since they are used to create dfs files
# Other geometries are available in the spatial module
from .spatial import (
    Grid1D,
    Grid2D,
    Grid3D,
)

from .xyz import read_xyz



def read(filename, *, items=None, time=None, keepdims=False, **kwargs) -> Dataset:
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
    error_bad_data: bool, optional
            raise error if data is corrupt, by default True,
    fill_bad_data_value:
            fill value for to impute corrupt data, used in conjunction with error_bad_data=False
            default np.nan

    Returns
    -------
    Dataset
        A Dataset with specification according to the file type

    See also
    --------
    mikeio.open - open a Dfs file and only read the header

    Examples
    --------
    >>> ds = mikeio.read("ts.dfs0")
    >>> ds = mikeio.read("ts.dfs0", items=0)
    >>> ds = mikeio.read("ts.dfs0", items="Temperature")
    >>> ds = mikeio.read("sw_points.dfs0, items="*Buoy 4*")
    >>> ds = mikeio.read("ts.dfs0", items=["u","v"], time="2016")
    >>> ds = mikeio.read("tide.dfs1", time="2018-5")
    >>> ds = mikeio.read("tide.dfs1", time=slice("2018-5-1","2018-6-1"))
    >>> ds = mikeio.read("tide.dfs1", items=[0,3,6], time=-1)
    >>> ds = mikeio.read("tide.dfs1", time=-1, keepdims=True)
    >>> ds = mikeio.read("era5.dfs2", area=(10,50,16,58))
    >>> ds = mikeio.read("HD2D.dfsu")
    >>> ds = mikeio.read("HD2D.dfsu", x=2.2, y=54.2)
    >>> ds = mikeio.read("HD2D.dfsu", elements=183)
    >>> ds = mikeio.read("HD2D.dfsu", elements=range(0,2000))
    >>> ds = mikeio.read("HD2D.dfsu", area=(10,50,16,58))
    >>> ds = mikeio.read("MT3D_sigma_z.dfsu", x=11.4, y=56.2)
    >>> ds = mikeio.read("MT3D_sigma_z.dfsu", x=11.4, y=56.2, z=-1.1)
    >>> ds = mikeio.read("MT3D_sigma_z.dfsu", elements=lst_of_elems)
    >>> ds = mikeio.read("MT3D_sigma_z.dfsu", layers="bottom")
    >>> ds = mikeio.read("MT3D_sigma_z.dfsu", layers=[-2,-1])
    >>> ds = mikeio.read("HD2D.dfsu", error_bad_data=False) # replace corrupt data with np.nan
    >>> ds = mikeio.read("HD2D.dfsu", error_bad_data=False, fill_bad_data_value=0.0) # replace corrupt data with 0.0
    """

    ext = Path(filename).suffix.lower()

    if "dfs" not in ext:
        raise ValueError("mikeio.read() is only supported for Dfs files")

    dfs = open(filename)

    return dfs.read(items=items, time=time, keepdims=keepdims, **kwargs)


def open(filename: str, **kwargs):
    """Open a dfs/mesh file (and read the header)

    The typical workflow for small dfs files is to read all data
    with *mikeio.read* instead of using this function. For big files, however,
    it can be convenient to open the file first with *dfs=mikeio.open(...)* to
    inspect it's content (items, time and shape) and then decide what to
    read using dfs.read(...)

    Parameters
    ----------
    filename
        full path and file name to the dfs file.
    type : str, optional
        Dfs2 only. Additional information about the file, e.g.
        "spectral" for spectral dfs2 files. By default: None.

    See also
    --------
    mikeio.read - read data from a dfs file

    Examples
    --------
    >>> dfs = mikeio.open("wl.dfs1")
    >>> dfs = mikeio.open("HD2D.dfsu")
    >>> ds = dfs.read(items="Salinity", time="2016-01")

    >>> dfs = mikeio.open("pt_spectra.dfs2", type="spectral")
    """
    ext = Path(filename).suffix.lower()[1:]

    READERS = {
        "dfs0": Dfs0,
        "dfs1": Dfs1,
        "dfs2": Dfs2,
        "dfs3": Dfs3,
        "dfsu": Dfsu,
        "mesh": Mesh,
    }

    if ext not in READERS:
        valid_formats = ", ".join(READERS.keys())
        raise Exception(
            f"{ext} is not a supported format for mikeio.open. Valid formats are {valid_formats}"
        )

    reader_klass = READERS[ext]

    return reader_klass(filename, **kwargs)

__all__ = [
    "DataArray",
    "Dataset",
    "Dfs0",
    "Dfs1",
    "Dfs2",
    "Dfs3",
    "Dfsu",
    "Mesh",
    "EUMType",
    "EUMUnit",
    "ItemInfo",
    "Pfs",
    "PfsDocument",
    "PfsSection",
    "read_pfs",
    "Grid1D",
    "Grid2D",
    "Grid3D",
    "read_xyz",
    "read",
    "open",
]