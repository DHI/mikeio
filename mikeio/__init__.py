import clr
import sys
import os
import platform


# sys.path.append(r"C:\Program Files (x86)\DHI\2019\bin\x64")
# sys.path.append(r"C:\Program Files (x86)\DHI\2020\bin\x64")
dirname = os.path.dirname(__file__)
mikebin = os.path.join(dirname, "bin")
sys.path.append(mikebin)
clr.AddReference("DHI.Generic.MikeZero.DFS")
clr.AddReference("DHI.Generic.MikeZero.EUM")
clr.AddReference("DHI.Projections")
clr.AddReference("System")
clr.AddReference("System.Runtime.InteropServices")
clr.AddReference("System.Runtime")

p = platform.architecture()
x64 = False
x64 = "64" in p[0]

if not x64:
    print("This library has not been tested in a 32 bit system!!!!")


from .dfs0 import Dfs0
from .dfs1 import Dfs1
from .dfs2 import Dfs2
from .dfs3 import Dfs3
from .dfsu import Dfsu
from .mesh import Mesh


def read(filename, item_numbers=None, item_names=None):
    """Read data from a dfs file

    Usage:
        read(filename, item_numbers=None, item_names=None)
    filename
        full path and file name to the dfs file.
    item_numbers
        read only the item_numbers in the array specified (0 base)
    item_names
        read only the items in the array specified, (takes precedence over item_numbers)

    Return:
        Dataset(data, time, names)
    """

    _, ext = os.path.splitext(filename)

    if ext == ".dfs0":

        dfs = Dfs0()

    elif ext == ".dfs1":

        dfs = Dfs1()

    elif ext == ".dfs2":

        dfs = Dfs2()

    elif ext == ".dfsu":

        dfs = Dfsu()
    else:
        raise Exception(f"{ext} is an unsupported extension")

    return dfs.read(filename, item_numbers, item_names)
