from pydhi import *

class dfs1():


    def read(self, dfs1file, item_numbers=None):
        """ Function: Read a dfs1 file

        usage:
            [data, time, name] = read(filename, item_numbers)
            item_numbers is a list of indices (base 0) to read from

        Returns
            1) the data contained in a dfs1 file in a list of numpy matrices
            2) time index
            3) name of the items

        NOTE
            Returns data (x, nt)
        """

        # NOTE. Item numbers are base 0 (everything else in the dfs is base 0)

        # Open the dfs file for reading
        dfs = DfsFileFactory.DfsGenericOpen(dfs1file)

        if item_numbers is None:
            item_numbers = list(range(dfs.ItemInfo.Count))


        # Determine the size of the grid
        axis = dfs.ItemInfo[0].SpatialAxis

        xNum = axis.XCount
        nt = dfs.FileInfo.TimeAxis.NumberOfTimeSteps
        if nt == 0:
            raise Warning("Static dfs1 files (with no time steps) are not supported.")
            nt = 1
        deleteValue = dfs.FileInfo.DeleteValueFloat

        n_items = len(item_numbers)
        data_list = []

        for item in range(n_items):
            # Initialize an empty data block
            data = np.ndarray(shape=(xNum, nt), dtype=float)
            data_list.append(data)

        t = []
        startTime = dfs.FileInfo.TimeAxis.StartDateTime;
        for it in range(dfs.FileInfo.TimeAxis.NumberOfTimeSteps):
            for item in range(n_items):

                itemdata = dfs.ReadItemTimeStep(item_numbers[item] + 1, it)

                if x64:
                    src = itemdata.Data
                    src_hndl = GCHandle.Alloc(src, GCHandleType.Pinned)
                    try:
                        src_ptr = src_hndl.AddrOfPinnedObject().ToInt64()
                        bufType = ctypes.c_float * len(src)
                        cbuf = bufType.from_address(src_ptr)
                        d = np.frombuffer(cbuf, dtype=cbuf._type_)
                    finally:
                        if src_hndl.IsAllocated: src_hndl.Free()

                else:
                    raise Warning("Slow read if using 32 bit Python.")
                    d = np.array(list(itemdata.Data))

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


    def write(self, dfs1file, data):
        """
        Function: write to a pre-created dfs1 file.

        dfs1file:
            full path and filename to existing dfs1 file

        data:
            list of matrices. len(data) must equal the number of items in the dfs2.
            Easch matrix must be of dimension x,time

        usage:
            write(filename, data) where  data(x, nt)

        Returns:
            Nothing

        """

        # Open the dfs file for writing
        dfs = DfsFileFactory.Dfs1FileOpenEdit(dfs1file)

        # Determine the size of the grid
        number_x = dfs.SpatialAxis.XCount
        n_time_steps = dfs.FileInfo.TimeAxis.NumberOfTimeSteps
        n_items = dfs.ItemInfo.Count

        deletevalue = -1e-035

        if not all(np.shape(d)[0] == number_x for d in data):
            raise Warning("ERROR data matrices in the X dimension do not all match in the data list. "
                     "Data is list of matices [x,time]")
        if not all(np.shape(d)[1] == n_time_steps for d in data):
            raise Warning("ERROR data matrices in the time dimension do not all match in the data list. "
                     "Data is list of matices [x,time]")
        if not len(data) == n_items:
            raise Warning("The number of matrices in data do not match the number of items in the dfs1 file.")

        for it in range(n_time_steps):
            for item in range(n_items):
                d = data[item][:, it]
                d[np.isnan(d)] = deletevalue
                #d = d.reshape(number_y, number_x)
                #d = np.flipud(d)
                darray = Array[System.Single](np.array(d.reshape(d.size, 1)[:, 0]))
                dfs.WriteItemTimeStepNext(0, darray)

        dfs.Close()
