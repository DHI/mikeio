import os
import numpy as np
from datetime import datetime, timedelta
from DHI.Generic.MikeZero import eumUnit, eumQuantity
from DHI.Generic.MikeZero.DFS import DfsFileFactory, DfsFactory
from DHI.Generic.MikeZero.DFS.dfsu import DfsuFile, DfsuFileType, DfsuBuilder, DfsuUtil
from DHI.Generic.MikeZero.DFS.mesh import MeshFile

from .dutil import Dataset, find_item, get_item_info
from .dotnet import (
    to_numpy,
    to_dotnet_float_array,
    to_dotnet_datetime,
    from_dotnet_datetime,
    asNumpyArray,
    to_dotnet_array
)
from .eum import TimeStep, ItemInfo
from .helpers import safe_length

class _Unstructured:
    _filename = None
    _source = None
    _filetype = None
    _projstr = None
    _nc = None
    _ec = None
    _valid_codes = None
    _start_time = None
    _items = None

    def __init__(self, filename):       
        self._filename = filename
        self._read_header(filename)

    def _read_header(self, filename):
        _, ext = os.path.splitext(filename)

        if ext == ".mesh":
            self._read_mesh_header(filename)            
        
        elif ext == ".dfsu":
            self._read_dfsu_header(filename)
            self._source.Close()

    def _read_mesh_header(self, filename):
        msh = MeshFile.ReadMesh(filename)
        self._source = msh
        self._projstr = msh.ProjectionString
        self._filetype = -1        

    def _read_dfsu_header(self, filename):
        
        dfs = DfsuFile.Open(filename)
        self._source = dfs
        self._projstr = dfs.Projection.WKTString
        self._filetype = dfs.DfsuFileType    
        self._deleteValue = dfs.DeleteValueFloat

        # items 
        self._n_items = safe_length(dfs.ItemInfo)
        self._items = get_item_info(dfs, list(range(self._n_items)))

        # time
        self._start_time = from_dotnet_datetime(dfs.StartDateTime)
        self._n_timesteps = dfs.NumberOfTimeSteps
        self._timestep_in_seconds = dfs.TimeStepInSeconds

        #dfs.Close()

    @property
    def node_coordinates(self):  
        if self._nc is None:
            #xn = np.array(list())
            xn = asNumpyArray(self._source.X)
            yn = asNumpyArray(self._source.Y)
            zn = asNumpyArray(self._source.Z)
            #yn = np.array(list(self._source.Y))
            #zn = np.array(list(self._source.Z))
            self._nc = np.column_stack([xn, yn, zn])      
        return self._nc

    @property
    def n_elements(self):
        return self._source.NumberOfElements

    @property
    def projection_string(self):
        return self._projstr

    @property
    def is_geo(self):
        return self._projstr == "LONG/LAT"
    
    @property
    def is_local_coordinates(self):
        return self._projstr == "NON-UTM"

    @property
    def valid_codes(self):
        if self._valid_codes is None:
            codes = np.array(list(self._source.Code))
            #codes = to_numpy(self._source.Code).astype(int)            
            self._valid_codes = list(set(codes))
        return self._valid_codes

    @property
    def boundary_codes(self):
        """provides a unique list of boundary codes
        """        
        return [code for code in self.valid_codes if code > 0]

    @property
    def element_coordinates(self):
        if self._ec is None:
            xc = np.zeros(self.n_elements)
            yc = np.zeros(self.n_elements)
            zc = np.zeros(self.n_elements)
            _, xc2, yc2, zc2 = DfsuUtil.CalculateElementCenterCoordinates(self._source, to_dotnet_array(xc), to_dotnet_array(yc), to_dotnet_array(zc))
            self._ec = np.column_stack([asNumpyArray(xc2), asNumpyArray(yc2), asNumpyArray(zc2)])
        return self._ec


class Dfsu(_Unstructured):
    
    def __init__(self, filename):
        super().__init__(filename)  # super(Dfsu, self)

    @property 
    def deletevalue(self):
        return self._deletevalue

    @property 
    def n_items(self):
        return self._n_items

    @property 
    def items(self):
        return self._items

    @property
    def start_time(self):
        return self._start_time

    @property
    def n_timesteps(self):
        return self._n_timesteps

    @property
    def timestep(self):
        return self._timestep_in_seconds

    @timestep.setter
    def timestep(self, value):
        if value <= 0:
            print(f'timestep must be positive scalar!')
        else:
            self._timestep_in_seconds = value

    @property
    def end_time(self):
        return self.start_time + timedelta(self.n_timesteps * self.timestep)


    # 3D dfsu stuff

    @property
    def n_layers(self):
        return self._source.NumberOfLayers

    @property
    def n_sigma_layers(self):
        return self._source.NumberOfSigmaLayers

    _top_elems = None

    @property 
    def top_element_ids(self):
        if self._top_elems is None:
            self._top_elems = np.array(DfsuUtil.FindTopLayerElements(self._source))
        return self._top_elems
    
    _n_layers_column = None

    @property 
    def num_layers_per_column(self):
        if self._n_layers_column is None:                    
            top_elems = self.top_element_ids
            n = len(top_elems)
            tmp = top_elems.copy()
            tmp[0] = -1
            tmp[1:n] = top_elems[0:(n-1)]
            self._n_layers_column = top_elems - tmp
        return self._n_layers_column

    _bot_elems = None

    @property 
    def bottom_element_ids(self):
        if self._bot_elems is None:
            self._bot_elems = self.top_element_ids - self.num_layers_per_column
        return self._bot_elems

    def get_element_ids_layer_n(self, n):
        """3D element ids for a specific layer

        Parameters
        ----------
        n : int
            layer between 1 (bottom) and n_layers (top) 
            (can also be negative with 0 as top layer )

        Returns
        -------
        np.array(int)
            element ids
        """
        n_lay = self.n_layers
        n_sigma = self.n_sigma_layers
        n_z = n_lay - n_sigma
        if n > n_z:
            n = n - n_lay

        if n < (-n_lay) or n > n_lay:
            print(f'Layer {n} not allowed must be between -{n_lay} and {n_lay}')
            raise Exception         
        if n <= 0:
            # sigma layers, counting from the top
            if n < -n_sigma:
                raise Exception(f'Negative layers only possible for sigma layers')
            return self.top_element_ids + n 
        else:
            # then it must be a z layer 
            return self.bottom_element_ids[self.num_layers_per_column >= n] + n 

    def get_nodes_for_elements(self, element_ids, node_layers = 'all'): 
        """list of unique node ids for a list of elements

        Parameters
        ----------
        element_ids : np.array(int)
            array of element ids
        node_layers : str, optional
            for 3D files 'all', 'bottom' or 'top' nodes of each element, by default 'all'

        Returns
        -------
        np.array(int)
            array of node ids (unique)
        """
        nodes = []        
        if (node_layers is None) or (node_layers == 'all') or self._filetype <= 0:
            for j in element_ids:
                elem_nodes = list(self._source.ElementTable[j])
                for node in elem_nodes:
                    nodes.append(node)
        else: 
            # 3D file    
            if (node_layers != 'bottom') and (node_layers != 'top'):
                raise Exception('node_layers must be either all, bottom or top')
            for j in element_ids:
                elem_nodes = list(self._source.ElementTable[j])
                nn = len(elem_nodes)
                halfn = int(nn/2)
                if (node_layers == 'bottom'):
                    elem_nodes = elem_nodes[:halfn]
                if (node_layers == 'top'):
                    elem_nodes = elem_nodes[halfn:]
                for node in elem_nodes:
                    nodes.append(node)    

        return np.unique(nodes)
        

    def read(self, items=None, time_steps=None):
        """
        Read data from a dfsu file

        Parameters
        ---------
        filename: str
            dfsu filename
        items: list[int] or list[str], optional
            Read only selected items, by number (0-based), or by name
        time_steps: list[int], optional
            Read only selected time_steps

        Returns
        -------
        Dataset
            A dataset with data dimensions [t,elements]
        """

        # Open the dfs file for reading
        #dfs = DfsuFile.Open(self._filename)
        self._read_dfsu_header(self._filename)
        dfs = self._source
        
        # NOTE. Item numbers are base 0 (everything else in the dfs is base 0)
        n_items = self.n_items #safe_length(dfs.ItemInfo)

        nt = dfs.NumberOfTimeSteps

        if items is not None and isinstance(items[0],str):
            items = find_item(dfs, items)

        if items is None:
            item_numbers = list(range(n_items))
        else:
            item_numbers = items
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
                    item_numbers[item] + 1, it
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

    
    def write(
        self,
        filename,
        data,
        start_time=None,
        dt=None,
        items=None,
        title=None,
    ):
        """Create a new dfsu file

        Parameters
        -----------
        filename: str
            full path to the new dfsu file
        data: list[np.array] or Dataset
            list of matrices, one for each item. Matrix dimension: time, x
        start_time: datetime, optional
            start datetime, default is datetime.now()
        dt: float, optional
            The time step (in seconds)
        items: list[ItemInfo], optional
        title: str
            title of the dfsu file. Default is blank.
        """

        if isinstance(data,Dataset):
            items = data.items
            start_time = data.time[0]
            if dt is None and len(data.time) > 1:
                dt = (data.time[1] - data.time[0]).total_seconds()
            data = data.data

        n_items = len(data)
        n_time_steps = np.shape(data[0])[0]

        if dt is None:
            dt = 1 # Arbitrary if there is only a single timestep

        if start_time is None:
            start_time = datetime.now()

        if items is None:
            items = [ItemInfo(f"Item {i+1}") for i in range(n_items)]

        if title is None:
            title = ""

        system_start_time = to_dotnet_datetime(start_time)

        # Default filetype;
        filetype = DfsuFileType.Dfsu2D
        
        _, ext = os.path.splitext(self._filename)

        if ext == ".mesh":

            source = MeshFile.ReadMesh(self._filename)
            projstr = source.ProjectionString

        elif ext == ".dfsu":

            source = DfsuFile.Open(self._filename)
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
        builder.SetNodeIds(source.NodeIds)
        builder.SetElementIds(source.ElementIds)

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
        nc = self.node_coordinates

        if code is not None:
            if code not in self.valid_codes:
                print(f"Selected code: {code} is not valid. Valid codes: {valid_codes}")
                raise Exception
            c = np.array(list(self._source.Code))
            return nc[c == code]

        return nc

    def get_element_coords(self):
        """FOR BACKWARD COMPATIBILITY ONLY. Use element_coordinates instead.
        """
        return self.element_coordinates

    def get_number_of_time_steps(self):
        """FOR BACKWARD COMPATIBILITY ONLY. Use n_timesteps instead.
        """
        return self.n_timesteps

    def find_n_closest_element_index(self, x, y, z=None, n=1):
        ec = self.element_coordinates

        if z is None:
            poi = np.array([x, y])

            d = ((ec[:, 0:2] - poi) ** 2).sum(axis=1)
            idx = d.argsort()[0:n]
        else:
            poi = np.array([x, y, z])

            d = ((ec - poi) ** 2).sum(axis=1)
            idx = d.argsort()[0:n]
        if n == 1:
            idx = idx[0]
        return idx

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
        if np.isscalar(x):
            return self.find_n_closest_element_index(x, y, z, n=1)
        else:
            nx = len(x)
            ny = len(y)
            if nx != ny:
                print(f"x and y must have same length")
                raise Exception
            idx = np.zeros(nx, dtype=int)
            if z is None:
                for j in range(nx):
                    idx[j] = self.find_n_closest_element_index(x[j], y[j], z=None, n=1)
            else: 
                nz = len(z)
                if nx != nz:
                    print(f"z must have same length as x and y")
                for j in range(nx):
                    idx[j] = self.find_n_closest_element_index(x[j], y[j], z[j], n=1)
        return idx

    def get_element_area(self):
        """Calculate the horizontal area of each element.

        Returns:
            np.array
                areas in m2
        """
        n_elements = self._source.NumberOfElements

        # Node coordinates
        xn = np.array(list(self._source.X))
        yn = np.array(list(self._source.Y))

        area = np.empty(n_elements)
        xcoords = np.empty(4)
        ycoords = np.empty(4)

        for j in range(n_elements):
            nodes = self._source.ElementTable[j]

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
