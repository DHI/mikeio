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
__version__ = "0.12.a0"

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


def read(filename, items=None, time_steps=None):
    """Read data from a dfs file

    Parameters
    ----------
    filename
        full path and file name to the dfs file.
    items: list[int] or list[str], optional
        Read only selected items, by number (0-based), or by name
    time_steps: int or list[int], optional
        Read only selected time_steps

    Returns
    -------
    Dataset
        A Dataset with data dimensions according to the file type
    """

    _, ext = os.path.splitext(filename)

    if ext == ".xyz":
        return read_xyz(filename)
    elif ext == ".mesh":
        # or should Mesh.read() just return geometry?
        raise Exception(f"{ext} cannot be read(). Try open() instead.")
    else:
        dfs = open(filename)

    return dfs.read(items, time_steps)


def open(filename: str):
    _, ext = os.path.splitext(filename)

    if ext == ".dfs0":
        return Dfs0(filename)

    elif ext == ".dfs1":
        return Dfs1(filename)

    elif ext == ".dfs2":
        return Dfs2(filename)

    elif ext == ".dfs3":
        return Dfs3(filename)

    elif ext == ".dfsu":
        return Dfsu(filename)

    elif ext == ".mesh":
        return Mesh(filename)

    else:
        raise Exception(f"{ext} is an unsupported extension")
