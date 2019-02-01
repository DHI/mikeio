import clr
from os import path
import pandas as pd
import numpy as np
import array
import sys
import System
from System import Array
import datetime
import ctypes
from ctypes import string_at
import sys


clr.AddReference("DHI.Generic.MikeZero.DFS");
clr.AddReference("DHI.Generic.MikeZero.EUM");
clr.AddReference("System");
clr.AddReference("System.Runtime.InteropServices")
clr.AddReference("System.Runtime")
from DHI.Generic.MikeZero import *
from DHI.Generic.MikeZero.DFS import *
from DHI.Generic.MikeZero.DFS.dfs0 import *
from DHI.Generic.MikeZero.DFS.dfs123 import *
from System.Runtime.InteropServices import GCHandle, GCHandleType

import platform
p=platform.architecture()
x64 = False
x64 = '64' in p[0]
if not x64:
    print('This library has not been tested in a 32 bit system!!!!')

from pydhi import dfs0 as dfs0
from pydhi import dfs2 as dfs2
from pydhi import dfs3 as dfs3
from pydhi import dfs_util as dfs_util