import clr
import sys
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
x64 = '64' in p[0]
if not x64:
    print('This library has not been tested in a 32 bit system!!!!')
