import os
import warnings
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
    to_dotnet_array,
    asnetarray_v2
)
from .eum import TimeStep, ItemInfo
from .helpers import safe_length

class _UnstructuredGeometry:
    # THIS CLASS KNOWS NOTHING ABOUT MIKE FILES!
    _projstr = None
    _n_nodes = None
    _n_elements = None     
    _nc = None
    _ec = None
    _codes = None
    _valid_codes = None
    _element_ids = None
    _node_ids = None
    _element_table = None
    _type = None    # -1: mesh, 0: 2d-dfsu, 4:dfsu3dsigma, ...

    def __repr__(self):
        out = []
        out.append("Unstructured Geometry")
        if self.n_nodes:
            out.append(f"Number of nodes: {self.n_nodes}")
        if self.n_elements:
            out.append(f"Number of elements: {self.n_elements}")
        if self._projstr:
            out.append(f"Projection: {self.projection_string}")
        return str.join("\n", out)
    
    @property
    def n_nodes(self):        
        return self._n_nodes

    @property
    def n_elements(self):        
        return self._n_elements

    @property
    def node_coordinates(self):      
        return self._nc

    @property
    def node_ids(self):        
        return self._node_ids

    @property
    def element_ids(self):        
        return self._element_ids

    @property
    def codes(self):
        return self._codes

    @property
    def valid_codes(self):
        if self._valid_codes is None:         
            self._valid_codes = list(set(self.codes))
        return self._valid_codes

    @property
    def boundary_codes(self):
        """provides a unique list of boundary codes
        """        
        return [code for code in self.valid_codes if code > 0]

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
            return nc[self.codes == code]

        return nc

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
    def element_table(self):
        return self._element_table

    @property 
    def max_nodes_per_element(self):
        maxnodes = 0
        for local_nodes in self.element_table:
            n = len(local_nodes)
            if n > maxnodes:
                maxnodes = n
        return maxnodes

    @property 
    def is_2d(self):
        return _type <= 0

    @property 
    def type_as_string(self):        
        """
        0: Dfsu2D: 2D area series
        1: DfsuVerticalColumn: 1D vertical column
        2: DfsuVerticalProfileSigma: 2D vertical slice through a Dfsu3DSigma
        3: DfsuVerticalProfileSigmaZ: 2D vertical slice through a Dfsu3DSigmaZ
        4: Dfsu3DSigma: 3D file with sigma coordinates, i.e., a constant number of layers.
        5: Dfsu3DSigmaZ: 3D file with sigma and Z coordinates, i.e. a varying number of layers.
        """
        if self._type == -1:
            return 'Mesh'
        if self._type == 0:
            return 'Dfsu2D'
        if self._type == 1:
            return 'DfsuVerticalColumn'
        if self._type == 2:
            return 'DfsuVerticalProfileSigma'
        if self._type == 3:
            return 'DfsuVerticalProfileSigmaZ'
        if self._type == 4:
            return 'Dfsu3DSigma'
        if self._type == 5:
            return 'Dfsu3DSigmaZ'
        return None

    def set_nodes(self, node_coordinates, codes=None, node_ids=None, projection_string=None):
        self._nc = np.asarray(node_coordinates)
        if codes is None:
            codes = np.zeros(len(node_coordinates), dtype=int)
        self._codes = np.asarray(codes)
        self._n_nodes = len(codes)
        if node_ids is None:
            node_ids = list(range(1,self._n_nodes+1))
        self._node_ids = np.asarray(node_ids)
        if projection_string is None:
            projection_string = "LONG/LAT"
        self._projstr = projection_string

    def set_elements(self, element_table, element_ids=None, geometry_type=None):
        self._element_table = element_table
        self._n_elements = len(element_table)
        if element_ids is None:
            element_ids = list(range(1,self.n_elements+1))
        self._element_ids = np.asarray(element_ids)
        
        if geometry_type is None:
            # guess type
            if self.max_nodes_per_element < 5:
                geometry_type = 0
            else:
                geometry_type = 4
        self._type = geometry_type

    def reindex(self):
        new_node_ids = range(1,self.n_nodes+1)
        new_element_ids = range(1,self.n_elements+1)
        node_dict = dict(zip(self.node_ids, new_node_ids))
        #print(node_dict)
        for j in range(self.n_elements):
            elem_nodes = self._element_table[j]
            #print(elem_nodes)
            new_elem_nodes = []
            for idx in elem_nodes:
                new_elem_nodes.append(node_dict[idx])
            self._element_table[j] = new_elem_nodes
            
        self._node_ids = list(new_node_ids)
        self._element_ids = list(new_element_ids)

    def get_element_table_for_elements(self, element_ids):
        elem_tbl = []        
        for j in element_ids:
            elem_nodes = self.element_table[j-1]
            elem_tbl.append(elem_nodes)  
        return elem_tbl

    def elements_to_geometry(self, elements):
        geom = _UnstructuredGeometry()
        node_ids = self.get_node_ids_for_elements(elements)
        node_idx = node_ids-1  # index is zero based
        #node_ids = self.node_ids[nodes]
        #print(node_ids)
        node_coords = self.node_coordinates[node_idx]
        codes = self.codes[node_idx]
        elem_tbl = self.get_element_table_for_elements(elements)
        #print(elem_tbl)
        geom.set_nodes(node_coords, codes=codes, node_ids=node_ids, projection_string=self.projection_string)
        geom.set_elements(elem_tbl, self.element_ids[elements])
        
        geom.reindex()
        #print(geomnode_ids)
        return geom

    def get_node_ids_for_elements(self, element_ids, node_layers = 'all'): 
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
        if (node_layers is None) or (node_layers == 'all') or self._type <= 0:
            for j in element_ids:
                elem_nodes = self.element_table[j-1] 
                for node in elem_nodes:
                    nodes.append(node)
        else: 
            # 3D file    
            if (node_layers != 'bottom') and (node_layers != 'top'):
                raise Exception('node_layers must be either all, bottom or top')
            for j in element_ids:
                elem_nodes = self.element_table[j-1] 
                nn = len(elem_nodes)
                halfn = int(nn/2)
                if (node_layers == 'bottom'):
                    elem_nodes = elem_nodes[:halfn]
                if (node_layers == 'top'):
                    elem_nodes = elem_nodes[halfn:]
                for node in elem_nodes:
                    nodes.append(node)    

        return np.unique(nodes)


    def validate(self):
        """ validate consistency of this mesh geometry
        """
        return False
    
    @property 
    def element_coordinates(self):
        if self._ec is None:
            self._ec = self.get_element_coords()
        return self._ec

    def get_element_coords(self):
        """Calculates the coordinates of the center of each element.
        Returns
        -------
            np.array
                x,y,z of each element
        """
        n_elements = self.n_elements

        ec = np.empty([n_elements, 3])

        # pre-allocate for speed
        maxnodes = self.max_nodes_per_element
        idx = np.zeros(maxnodes, dtype=np.int)
        xcoords = np.zeros([maxnodes, n_elements])
        ycoords = np.zeros([maxnodes, n_elements])
        zcoords = np.zeros([maxnodes, n_elements])
        nnodes_per_elem = np.zeros(n_elements)

        for j in range(n_elements):
            nodes = self._element_table[j]
            nnodes = len(nodes)
            nnodes_per_elem[j] = nnodes
            for i in range(nnodes):
                idx[i] = nodes[i] - 1

            xcoords[:nnodes,j] = self._nc[idx[:nnodes],0]
            ycoords[:nnodes,j] = self._nc[idx[:nnodes],1]
            zcoords[:nnodes,j] = self._nc[idx[:nnodes],2]
        
        ec[:, 0] = np.sum(xcoords, axis=0)/nnodes_per_elem
        ec[:, 1] = np.sum(ycoords, axis=0)/nnodes_per_elem
        ec[:, 2] = np.sum(zcoords, axis=0)/nnodes_per_elem

        self._ec = ec
        return ec

    def to_polygons(self):
        from matplotlib.patches import Polygon
        polygons = []

        for j in range(self.n_elements):
            nodes = self.element_table[j]
            pcoords = np.empty([len(nodes), 2])
            for i in range(len(nodes)):
                nidx = nodes[i] - 1
                pcoords[i, :] = self.node_coordinates[nidx, 0:2]

            polygon = Polygon(pcoords, True)
            polygons.append(polygon)   
        return polygons
    
    def to_shapely(self):
        from shapely.geometry import Polygon, MultiPolygon

        polygons = []
        for j in range(self.n_elements):
            nodes = self.element_table[j]
            pcoords = np.empty([len(nodes), 2])
            for i in range(len(nodes)):
                nidx = nodes[i] - 1
                pcoords[i, :] = self.node_coordinates[nidx, 0:2]
            polygon = Polygon(pcoords)
            polygons.append(polygon)
        mp = MultiPolygon(polygons)

        return mp

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


class _UnstructuredFile(_UnstructuredGeometry):
    _filename = None
    _source = None
    _start_time = None
    _timestep_in_seconds = None
    _items = None

    def __repr__(self):
        out = []
        out.append("Dfsu/Mesh")
        out.append(f"Number of nodes: {self.n_nodes}")
        out.append(f"Number of elements: {self.n_elements}")
        return str.join("\n", out)

    def __init__(self):       
        super().__init__()
        #self._filename = filename
        #self._read_header(filename)

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
        self._type = -1

        # geometry
        self._set_nodes_from_source(msh)
        self._set_elements_from_source(msh)

    def _read_dfsu_header(self, filename):
        
        dfs = DfsuFile.Open(filename)
        self._source = dfs
        self._projstr = dfs.Projection.WKTString
        self._type = dfs.DfsuFileType    
        self._deleteValue = dfs.DeleteValueFloat

        # geometry
        self._set_nodes_from_source(dfs)
        self._set_elements_from_source(dfs)

        # items 
        self._n_items = safe_length(dfs.ItemInfo)
        self._items = get_item_info(dfs, list(range(self._n_items)))

        # time
        self._start_time = from_dotnet_datetime(dfs.StartDateTime)
        self._n_timesteps = dfs.NumberOfTimeSteps
        self._timestep_in_seconds = dfs.TimeStepInSeconds

        #dfs.Close()

    def _set_nodes_from_source(self, source):
        xn = asNumpyArray(source.X)
        yn = asNumpyArray(source.Y)
        zn = asNumpyArray(source.Z)
        self._nc = np.column_stack([xn, yn, zn])   
        self._codes = np.array(list(source.Code))
        self._n_nodes = source.NumberOfNodes
        self._node_ids = np.array(list(source.NodeIds))

    def _set_elements_from_source(self, source):        
        elem_tbl = []
        self._n_elements = source.NumberOfElements
        for j in range(self.n_elements):
            elem_nodes = list(source.ElementTable[j])
            elem_tbl.append(elem_nodes)
        self._element_table = elem_tbl
        self._element_ids = np.array(list(source.ElementIds))
        
    # @property
    # def node_coordinates(self):  
    #     if self._nc is None:
    #         xn = asNumpyArray(self._source.X)
    #         yn = asNumpyArray(self._source.Y)
    #         zn = asNumpyArray(self._source.Z)
    #         self._nc = np.column_stack([xn, yn, zn])      
    #     return self._nc

    # @property
    # def n_nodes(self):
    #     if self._n_nodes is None:
    #         self._n_nodes = self._source.NumberOfNodes
    #     return self._n_nodes

    # @property
    # def n_elements(self):
    #     if self._n_elements is None:
    #         self._n_elements = self._source.NumberOfElements
    #     return self._n_elements

    # @property
    # def codes(self):
    #     if self._codes is None:
    #         self._codes = np.array(list(self._source.Code))
    #     return self._codes

    # @property
    # def element_table(self):
    #     if self._element_table is None:
    #         elem_tbl = []        
    #         for j in range(self.n_elements):
    #             elem_nodes = list(self._source.ElementTable[j])
    #             elem_tbl.append(elem_nodes)
    #         self._element_table = elem_tbl
    #     return self._element_table


class Dfsu(_UnstructuredFile):
    
    def __init__(self, filename):
        super().__init__()
        #super().__init__(filename) 
        self._filename = filename
        self._read_header(filename)

    @property
    def element_coordinates(self):
        if self._ec is None:
            xc = np.zeros(self.n_elements)
            yc = np.zeros(self.n_elements)
            zc = np.zeros(self.n_elements)
            _, xc2, yc2, zc2 = DfsuUtil.CalculateElementCenterCoordinates(self._source, to_dotnet_array(xc), to_dotnet_array(yc), to_dotnet_array(zc))
            self._ec = np.column_stack([asNumpyArray(xc2), asNumpyArray(yc2), asNumpyArray(zc2)])
        return self._ec

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
        if n > n_z and n <= n_lay:
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
            return self.bottom_element_ids[self.num_layers_per_column >= (n_lay-n+1)] + n 

    def read(self, items=None, time_steps=None, element_ids=None):
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
        element_ids: list[int], optional
            Read only selected element ids   

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

        if element_ids is None:
            n_elems = self.n_elements
            n_nodes = self.n_nodes
        else:
            node_ids = self.get_node_ids_for_elements(element_ids)
            n_elems = len(element_ids)
            n_nodes = len(node_ids)

        deleteValue = dfs.DeleteValueFloat

        data_list = []

        items = get_item_info(dfs, item_numbers)

        item0_is_node_based = False
        for item in range(n_items):
            # Initialize an empty data block
            if item == 0 and items[item].name == "Z coordinate":
                item0_is_node_based = True
                data = np.ndarray(shape=(len(time_steps), n_nodes), dtype=float)
            else:
                data = np.ndarray(shape=(len(time_steps), n_elems), dtype=float)
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

                if element_ids is not None:
                    if item==0 and item0_is_node_based:
                        d = d[node_ids]
                    else:
                        d = d[element_ids]

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
        element_ids=None,
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
                if not data.is_equidistant:
                    raise Exception("Data is not equidistant in time. Dfsu requires equidistant temporal axis!")
                dt = (data.time[1] - data.time[0]).total_seconds()
            data = data.data

        n_items = len(data)
        n_time_steps = np.shape(data[0])[0]

        if dt is None:
            if self.timestep is None:
                dt = 1
            else:
                dt = self.timestep #1 # Arbitrary if there is only a single timestep

        if start_time is None:
            if self.start_time is None:
                start_time = datetime.now()
                warnings.warn(f"No start time supplied. Using current time: {start_time} as start time.")
            else:
                start_time = self.start_time 
                warnings.warn(f"No start time supplied. Using start time from source: {start_time} as start time.")

        if items is None:
            items = [ItemInfo(f"Item {i+1}") for i in range(n_items)]

        if title is None:
            title = ""

        file_start_time = to_dotnet_datetime(start_time)

        # Default filetype;
        if self._type == -1:
            # create dfs2d from mesh
            filetype = DfsuFileType.Dfsu2D
        else:
            # same as source
            # TODO: if subset is 2d or slice... 
            filetype = self._type
        
        if filetype != DfsuFileType.Dfsu2D:
            if items[0].name != "Z coordinate":
                raise Exception("First item must be z coordinates of the nodes!")  

        # spatial subset 
        if element_ids is None:
            geometry = self
        else:
            geometry = self.elements_to_geometry(element_ids)

        xn = geometry.node_coordinates[:,0]
        yn = geometry.node_coordinates[:,1]

        # zn have to be Single precision??
        zn = to_dotnet_float_array(geometry.node_coordinates[:,2])

        elementtable = asnetarray_v2(geometry.element_table)

        builder = DfsuBuilder.Create(filetype)

        builder.SetNodes(xn, yn, zn, geometry.codes)
        builder.SetElements(elementtable)
        builder.SetNodeIds(geometry.node_ids)
        builder.SetElementIds(geometry.element_ids)

        factory = DfsFactory()
        proj = factory.CreateProjection(geometry.projection_string)
        builder.SetProjection(proj)
        builder.SetTimeInfo(file_start_time, dt)
        builder.SetZUnit(eumUnit.eumUmeter)

        if filetype != DfsuFileType.Dfsu2D:
            # TODO: take this from geometry
            builder.SetNumberOfSigmaLayers(self._source.NumberOfSigmaLayers)
           
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

    def get_element_coords(self):
        """FOR BACKWARD COMPATIBILITY ONLY. Use element_coordinates instead.
        """
        return self.element_coordinates

    def get_number_of_time_steps(self):
        """FOR BACKWARD COMPATIBILITY ONLY. Use n_timesteps instead.
        """
        return self.n_timesteps
