import numpy as np
from datetime import datetime
import System
from System import Array
from DHI.Generic.MikeZero import eumUnit, eumQuantity
from DHI.Generic.MikeZero.DFS import (
    DfsFileFactory,
    DfsFactory,
    DfsSimpleType,
    DataValueType,
)
from DHI.Generic.MikeZero.DFS.dfsu import DfsuFile, DfsuFileType, DfsuBuilder
from DHI.Generic.MikeZero.DFS.mesh import MeshFile

from .dutil import to_numpy, Dataset, find_item
from .eum import TimeStep
from .helpers import safe_length


class Dfsu:
    def read(self, filename, item_numbers=None, item_names=None, time_steps=None):
        """ Function: Read a dfsu file

        usage:
            [data, time, name] = read(filename, item_numbers)
            item_numbers is a list of indices (base 0) to read from

        Returns
            1) the data contained in a dfsu file in a list of numpy matrices
            2) time index
            3) name of the items
        """

        # Open the dfs file for reading
        dfs = DfsuFile.Open(filename)
        self._dfs = dfs

        # NOTE. Item numbers are base 0 (everything else in the dfs is base 0)
        item_offset = 0
        n_items = safe_length(dfs.ItemInfo)

        # Dynamic Z is the first item in 3d files
        if (dfs.DfsuFileType == DfsuFileType.Dfsu3DSigma) or (
            dfs.DfsuFileType == DfsuFileType.Dfsu3DSigmaZ
        ):
            item_offset = 1
            n_items = n_items - 1

        nt = dfs.NumberOfTimeSteps

        if item_names is not None:
            item_numbers = find_item(dfs, item_names)

        if item_numbers is None:
            item_numbers = list(range(n_items))
        else:
            n_items = len(item_numbers)

        if time_steps is None:
            time_steps = list(range(nt))

        xNum = dfs.NumberOfElements

        deleteValue = dfs.DeleteValueFloat

        data_list = []

        for item in range(n_items):
            # Initialize an empty data block
            data = np.ndarray(shape=(len(time_steps), xNum), dtype=float)
            data_list.append(data)

        t = []
        startTime = dfs.StartDateTime

        for i in range(len(time_steps)):
            it = time_steps[i]
            for item in range(n_items):

                itemdata = dfs.ReadItemTimeStep(
                    item_numbers[item] + item_offset + 1, it
                )

                src = itemdata.Data

                d = to_numpy(src)

                d[d == deleteValue] = np.nan
                data_list[item][i, :] = d

            t.append(
                startTime.AddSeconds(itemdata.Time).ToString("yyyy-MM-dd HH:mm:ss")
            )

        time = [datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in t]

        names = []
        for item in range(n_items):
            name = dfs.ItemInfo[item_numbers[item] + item_offset].Name
            names.append(name)

        dfs.Close()
        return Dataset(data_list, time, names)

    def write(self, filename, data):
        """
        Function: write to a pre-created dfsu file.

        filename:
            full path and filename to existing dfsu file

        data:
            list of matrices. len(data) must equal the number of items in the dfsu.
            Each matrix must be of dimension time,elements

        usage:
            write(filename, data) where  data(nt,elements)

        Returns:
            Nothing

        """

        # Open the dfs file for writing
        dfs = DfsFileFactory.DfsGenericOpenEdit(filename)

        n_time_steps = dfs.FileInfo.TimeAxis.NumberOfTimeSteps
        n_items = safe_length(dfs.ItemInfo)

        deletevalue = dfs.FileInfo.DeleteValueFloat

        for i in range(n_time_steps):
            for item in range(n_items):
                d = data[item][i, :]
                d[np.isnan(d)] = deletevalue
                darray = Array[System.Single](np.array(d.reshape(d.size, 1)[:, 0]))
                dfs.WriteItemTimeStepNext(0, darray)

        dfs.Close()

    def create(
        self,
        meshfilename,
        filename,
        data,
        start_time=None,
        dt=1,
        timeseries_unit=TimeStep.SECOND,
        variable_type=None,
        unit=None,
        names=None,
        title=None,
    ):

        n_items = len(data)
        n_time_steps = np.shape(data[0])[0]

        if start_time is None:
            start_time = datetime.now()

        if names is None:
            names = [f"Item {i+1}" for i in range(n_items)]

        if variable_type is None:
            variable_type = [999] * n_items

        if unit is None:
            unit = [0] * n_items

        if title is None:
            title = ""

        system_start_time = System.DateTime(
            start_time.year,
            start_time.month,
            start_time.day,
            start_time.hour,
            start_time.minute,
            start_time.second,
        )

        mesh = MeshFile.ReadMesh(meshfilename)

        # TODO support all types of Dfsu
        builder = DfsuBuilder.Create(DfsuFileType.Dfsu2D)

        # Setup header and geometry, copy from source file

        # zn have to be Single precision??
        zn = Array[System.Single](list(mesh.Z))
        builder.SetNodes(mesh.X, mesh.Y, zn, mesh.Code)

        builder.SetElements(mesh.ElementTable)
        factory = DfsFactory()
        proj = factory.CreateProjection(mesh.ProjectionString)
        builder.SetProjection(proj)
        builder.SetTimeInfo(system_start_time, dt)
        builder.SetZUnit(eumUnit.eumUmeter)

        for i in range(n_items):
            builder.AddDynamicItem(
                names[i], eumQuantity.Create(variable_type[i], unit[i])
            )

        try:
            dfs = builder.CreateFile(filename)
        except IOError:
            print("cannot create dfsu file: ", filename)

        deletevalue = dfs.DeleteValueFloat

        # Add data for all item-timesteps, copying from source
        for i in range(n_time_steps):
            for item in range(n_items):
                d = data[item][i, :]
                d[np.isnan(d)] = deletevalue
                darray = Array[System.Single](np.array(d.reshape(d.size, 1)[:, 0]))
                dfs.WriteItemTimeStepNext(0, darray)

        dfs.Close()

    def get_node_coords(self, code=None):
        """
        Function: get node coordinates, optionally filtered by code, land==1
        """
        # Node coordinates
        xn = np.array(list(self._dfs.X))
        yn = np.array(list(self._dfs.Y))
        zn = np.array(list(self._dfs.Z))

        nc = np.column_stack([xn, yn, zn])

        if code is not None:

            c = np.array(list(self._dfs.Code))
            valid_codes = set(c)

            if code not in valid_codes:

                print(f"Selected code: {code} is not valid. Valid codes: {valid_codes}")
                raise Exception
            return nc[c == code]

        return nc

    def get_element_coords(self):
        """
        Function: Calculates the coordinates of the center of each element.

        usage:
            dfs.get_element_coords() where dfs = dfs.read(dfsufile)

        Returns:
            nd-array of x,y,z of each element
        """
        n_elements = self._dfs.NumberOfElements

        # Node coordinates
        xn = np.array(list(self._dfs.X))
        yn = np.array(list(self._dfs.Y))
        zn = np.array(list(self._dfs.Z))

        ec = np.empty([n_elements, 3])

        for j in range(n_elements):
            nodes = self._dfs.ElementTable[j]

            xcoords = np.empty(nodes.Length)
            ycoords = np.empty(nodes.Length)
            zcoords = np.empty(nodes.Length)
            for i in range(nodes.Length):
                nidx = nodes[i] - 1
                xcoords[i] = xn[nidx]
                ycoords[i] = yn[nidx]
                zcoords[i] = zn[nidx]

            ec[j, 0] = xcoords.mean()
            ec[j, 1] = ycoords.mean()
            ec[j, 2] = zcoords.mean()

        return ec

    def find_closest_element_index(self, x, y, z=None):

        ec = self.get_element_coords()

        if z is None:
            poi = np.array([x, y])

            d = ((ec[:, 0:2] - poi) ** 2).sum(axis=1)
            idx = d.argsort()[0]
        else:
            poi = np.array([x, y, z])

            d = ((ec - poi) ** 2).sum(axis=1)
            idx = d.argsort()[0]

        return idx

    def get_number_of_time_steps(self):
        return self._dfs.get_NumberOfTimeSteps()

    @property
    def is_geo(self):
        """
        Function: Determines if dfsu file is defined on geographical LONG/LAT mesh.

        usage:
            dfs.is_geo() where dfs = dfs.read(dfsufile)

        Returns:
            True if LONG/LAT, FALSE otherwise
        """
        return self._dfs.Projection.WKTString == "LONG/LAT"

    def get_element_area(self):
        """
        Function: Calculates the horizontal area of each element.

        usage:
            dfs.get_element_area() where dfs = dfs.read(dfsufile)

        Returns:
            array of areas in m2
        """
        n_elements = self._dfs.NumberOfElements
        # element_ids  = self._dfs.ElementIds

        # Node coordinates
        xn = np.array(list(self._dfs.X))
        yn = np.array(list(self._dfs.Y))

        area = np.empty(n_elements)
        xcoords = np.empty(4)
        ycoords = np.empty(4)

        for j in range(n_elements):
            nodes = self._dfs.ElementTable[j]

            for i in range(nodes.Length):
                nidx = nodes[i] - 1
                xcoords[i] = xn[nidx]
                ycoords[i] = yn[nidx]

            # ab : edge vector corner a to b
            abx = xcoords[1] - xcoords[0]
            aby = ycoords[1] - ycoords[0]

            # ac : edge vector corner a to c
            acx = xcoords[2] - xcoords[0]
            acy = ycoords[2] - ycoords[0]

            isquad = False
            if nodes.Length > 3:
                isquad = True
                # ad : edge vector corner a to d
                adx = xcoords[3] - xcoords[0]
                ady = ycoords[3] - ycoords[0]

            # if geographical coords, convert all length to meters
            if self.is_geo:
                earth_radius = 6366707.0
                deg_to_rad = np.pi / 180.0
                earth_radius_deg_to_rad = earth_radius * deg_to_rad

                # Y on element centers
                Ye = np.sum(ycoords[: nodes.Length]) / nodes.Length
                cosYe = np.cos(np.deg2rad(Ye))

                abx = earth_radius_deg_to_rad * abx * cosYe
                aby = earth_radius_deg_to_rad * aby
                acx = earth_radius_deg_to_rad * acx * cosYe
                acy = earth_radius_deg_to_rad * acy
                if isquad:
                    adx = earth_radius_deg_to_rad * adx * cosYe
                    ady = earth_radius_deg_to_rad * ady

            # calculate area in m2
            area[j] = 0.5 * (abx * acy - aby * acx)
            if isquad:
                area[j] = area[j] + 0.5 * (acx * ady - acy * adx)

        return np.abs(area)
