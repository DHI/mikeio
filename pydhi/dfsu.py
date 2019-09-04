import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import System
from System import Array
from DHI.Generic.MikeZero import eumUnit, eumQuantity
from DHI.Generic.MikeZero.DFS import DfsFileFactory, DfsFactory, DfsSimpleType, DataValueType
from DHI.Generic.MikeZero.DFS.dfsu import DfsuFile, DfsuFileType
from System.Runtime.InteropServices import GCHandle, GCHandleType
import ctypes

class dfsu():


    def read(self, dfsufile, item_numbers=None):
        """ Function: Read a dfsu file

        usage:
            [data, time, name] = read(filename, item_numbers)
            item_numbers is a list of indices (base 0) to read from

        Returns
            1) the data contained in a dfsu file in a list of numpy matrices
            2) time index
            3) name of the items

        NOTE
            Returns data (x, nt)
        """

        # NOTE. Item numbers are base 0 (everything else in the dfs is base 0)

        # Open the dfs file for reading
        dfs = DfsuFile.Open(dfsufile)
        self._dfs = dfs

        if item_numbers is None:
            item_numbers = list(range(dfs.ItemInfo.Count))


        xNum = dfs.NumberOfElements
        nt = dfs.NumberOfTimeSteps

        deleteValue = dfs.DeleteValueFloat

        n_items = len(item_numbers)
        data_list = []

        for item in range(n_items):
            # Initialize an empty data block
            data = np.ndarray(shape=(xNum, nt), dtype=float)
            data_list.append(data)

        t = []
        startTime = dfs.StartDateTime;
        for it in range(nt):
            for item in range(n_items):

                itemdata = dfs.ReadItemTimeStep(item_numbers[item] + 1, it)

                src = itemdata.Data
                src_hndl = GCHandle.Alloc(src, GCHandleType.Pinned)
                try:
                    src_ptr = src_hndl.AddrOfPinnedObject().ToInt64()
                    bufType = ctypes.c_float * len(src)
                    cbuf = bufType.from_address(src_ptr)
                    d = np.frombuffer(cbuf, dtype=cbuf._type_)
                finally:
                    if src_hndl.IsAllocated: src_hndl.Free()

                d[d == deleteValue] = np.nan
                data_list[item][:, it] = d

            t.append(startTime.AddSeconds(itemdata.Time).ToString("yyyy-MM-dd HH:mm:ss"))

        time = pd.DatetimeIndex(t)
        names = []
        for item in range(n_items):
            name = dfs.ItemInfo[item].Name
            names.append(name)

        dfs.Close()
        return (data_list, time, names)

    def _get_element_coords(self):
        ne = self._dfs.NumberOfElements

        # Node coordinates
        xn = np.array(list(self._dfs.X))
        yn = np.array(list(self._dfs.Y))

        ec = np.empty([ne,2])

        for j in range(ne):
            nodes = self._dfs.ElementTable[j]

            xcoords = np.empty(nodes.Length)
            ycoords = np.empty(nodes.Length)
            for i in range(nodes.Length):
                nidx = nodes[i]-1
                xcoords[i] = xn[nidx]
                ycoords[i] = yn[nidx]

            ec[j,0] = xcoords.mean()
            ec[j,1] = ycoords.mean()

        return ec

    def find_closest_element_index(self, x, y):

        ec = self._get_element_coords()
        poi = np.array([x,y])

        d = ((ec - poi)**2).sum(axis=1)
        idx = d.argsort()[0]

        return idx




