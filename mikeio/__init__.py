import clr
import sys
import os
import platform

sys.path.append(r"C:\Program Files (x86)\DHI\2019\bin\x64")
sys.path.append(r"C:\Program Files (x86)\DHI\2020\bin\x64")
clr.AddReference("DHI.Generic.MikeZero.DFS")
clr.AddReference("DHI.Generic.MikeZero.EUM")
clr.AddReference("System")
clr.AddReference("System.Runtime.InteropServices")
clr.AddReference("System.Runtime")

p = platform.architecture()
x64 = False
x64 = "64" in p[0]

if not x64:
    print("This library has not been tested in a 32 bit system!!!!")


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
        from .dfs0 import dfs0

        dfs = dfs0()

    elif ext == ".dfs1":
        from .dfs1 import dfs1

        dfs = dfs1()

    elif ext == ".dfs2":
        from .dfs2 import dfs2

        dfs = dfs2()

    elif ext == ".dfsu":
        from .dfsu import dfsu

        dfs = dfsu()
    else:
        raise Exception(f"{ext} is an unsupported extension")

    return dfs.read(filename, item_numbers, item_names)
