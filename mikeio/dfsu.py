import os
import numpy as np
from datetime import datetime, timedelta
from DHI.Generic.MikeZero import eumUnit, eumQuantity
from DHI.Generic.MikeZero.DFS import DfsFileFactory, DfsFactory
from DHI.Generic.MikeZero.DFS.dfsu import DfsuFile, DfsuFileType, DfsuBuilder
from DHI.Generic.MikeZero.DFS.mesh import MeshFile

from .dutil import Dataset, find_item, get_item_info
from .dotnet import (
    to_numpy,
    to_dotnet_float_array,
    to_dotnet_datetime,
    from_dotnet_datetime,
)
from .eum import TimeStep, ItemInfo
from .helpers import safe_length


class Dfsu:
    def read(self, filename, item_numbers=None, item_names=None, time_steps=None):
        """
        Read a dfsu file

        Parameters
        ---------
        filename: str
            dfsu filename
        item_numbers: list[int], optional
            Read only selected items, by number (0-based)
        item_names: list[str], optional
            Read only selected items, by name, takes precedence over item_numbers
        time_steps: list[int], optional
            Read only selected time_steps

        Returns
        -------
        Dataset
            A dataset with data dimensions [t,elements]
        """

        # Open the dfs file for reading
        dfs = DfsuFile.Open(filename)
        self._dfs = dfs

        # NOTE. Item numbers are base 0 (everything else in the dfs is base 0)
        item_offset = 0
        n_items = safe_length(dfs.ItemInfo)

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

        items = get_item_info(dfs, item_numbers)

        for item in range(n_items):
            # Initialize an empty data block
            if item == 0 and items[item].name == "Z coordinate":
                data = np.ndarray(shape=(len(time_steps), dfs.NumberOfNodes), dtype=float)
            else:
                data = np.ndarray(shape=(len(time_steps), xNum), dtype=float)
            data_list.append(data)

        t_seconds = np.zeros(len(time_steps), dtype=float)

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

            t_seconds[i] = itemdata.Time

        start_time = from_dotnet_datetime(dfs.StartDateTime)
        time = [start_time + timedelta(seconds=tsec) for tsec in t_seconds]

        

        dfs.Close()
        return Dataset(data_list, time, items)

    def write(self, filename, data):
        """Overwrite a pre-created dfsu file.

        Parameters
        ----------
        filename: str
            full path and filename to existing dfsu file
        data: list[np.array]
            list of matrices. len(data) must equal the number of items in the dfsu.
            Each matrix must be of dimension time,elements
        """

        # Open the dfs file for writing
        dfs = DfsFileFactory.DfsGenericOpenEdit(filename)

        n_time_steps = dfs.FileInfo.TimeAxis.NumberOfTimeSteps
        n_items = safe_length(dfs.ItemInfo)

        if len(data) != n_items:
            dfs.Close()
            raise ValueError(
                "Number of items in data must equal number of items in the file"
            )

        if data[0].shape[0] != n_time_steps:
            dfs.Close()
            raise ValueError(
                "Number of timesteps in data must equal number of timesteps in the file"
            )

        deletevalue = dfs.FileInfo.DeleteValueFloat

        for i in range(n_time_steps):
            for item in range(n_items):
                d = data[item][i, :]
                d[np.isnan(d)] = deletevalue
                darray = to_dotnet_float_array(d)
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
        items=None,
        title=None,
    ):
        """Create a dfsu file

        Parameters
        -----------
        meshfilename: str,
            full path to a mesh or dfsu file
        filename: str
            full path to the new dfsu file
        data: list[np.array]
            list of matrices, one for each item. Matrix dimension: time, y, x
        start_time: datetime, optional
            start datetime, default is datetime.now()
        dt: float
            The time step. Therefore dt of 5.5 with timeseries_unit of TimeStep.MINUTE
            means 5 mins and 30 seconds. Default 1
        timeseries_unit: TimeStep, optional
             default TimeStep.SECOND
        unit: list[ItemInfo], optional
            Name, item, unit, default is undefined
        title: str
            title of the dfsu file. Default is blank.
        """

        n_items = len(data)
        n_time_steps = np.shape(data[0])[0]

        if start_time is None:
            start_time = datetime.now()

        if items is None:
            items = [ItemInfo(f"temItem {i+1}") for i in range(n_items)]

        if title is None:
            title = ""

        system_start_time = to_dotnet_datetime(start_time)

        # Default filetype; TODO support all types of Dfsu
        filetype = DfsuFileType.Dfsu2D
        
        _, ext = os.path.splitext(meshfilename)

        if ext == ".mesh":

            source = MeshFile.ReadMesh(meshfilename)
            projstr = source.ProjectionString

        elif ext == ".dfsu":

            source = DfsuFile.Open(meshfilename)
            projstr = source.Projection.WKTString
            filetype = source.DfsuFileType

        xn = source.X
        yn = source.Y

        # zn have to be Single precision??
        zn = to_dotnet_float_array(np.array(list(source.Z)))

        nodecodes = source.Code
        elementtable = source.ElementTable

        builder = DfsuBuilder.Create(filetype)

        builder.SetNodes(xn, yn, zn, nodecodes)
        builder.SetElements(elementtable)
        #builder.SetNodeIds(source.NodeIds)
        #builder.SetElementIds(source.ElementIds)

        factory = DfsFactory()
        proj = factory.CreateProjection(projstr)
        builder.SetProjection(proj)
        builder.SetTimeInfo(system_start_time, dt)
        builder.SetZUnit(eumUnit.eumUmeter)

        if filetype != DfsuFileType.Dfsu2D:
            builder.SetNumberOfSigmaLayers(source.NumberOfSigmaLayers)
           
        for item in items:
            if item.name != "Z coordinate":
                builder.AddDynamicItem(item.name, eumQuantity.Create(item.type, item.unit))

        try:
            dfs = builder.CreateFile(filename)
        except IOError:
            print("cannot create dfsu file: ", filename)

        deletevalue = dfs.DeleteValueFloat

        try:
            # Add data for all item-timesteps, copying from source
            for i in range(n_time_steps):
                for item in range(n_items):
                    d = data[item][i, :]
                    d[np.isnan(d)] = deletevalue
                    darray = to_dotnet_float_array(d)
                    dfs.WriteItemTimeStepNext(0, darray)
            dfs.Close()

        except Exception as e:
            print(e)
            dfs.Close()
            os.remove(filename)

    def get_node_coords(self, code=None):
        """Get the coordinates of each node.


        Parameters
        ----------

        code: int
            Get only nodes with specific code, e.g. land == 1

        Returns
        -------
            np.array
                x,y,z of each node
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
        """Calculates the coordinates of the center of each element.

        Returns
        -------
            np.array
                x,y,z of each element
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
        """Find index of closest element

        Parameters
        ----------

        x: float
            X coordinate(easting or longitude)
        y: float
            Y coordinate(northing or latitude)
        z: float, optional
          Z coordinate(depth, positive upwards)
        """

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
        """Determines if dfsu file is defined on geographical LONG/LAT mesh.

        Returns:
            bool
                True if LONG/LAT, FALSE otherwise
        """
        return self._dfs.Projection.WKTString == "LONG/LAT"

    def get_element_area(self):
        """Calculate the horizontal area of each element.

        Returns:
            np.array
                areas in m2
        """
        n_elements = self._dfs.NumberOfElements

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
