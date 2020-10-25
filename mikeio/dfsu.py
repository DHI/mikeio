import os
from enum import IntEnum
from collections import namedtuple 
import warnings
import numpy as np
from datetime import datetime, timedelta
from scipy.spatial import cKDTree 

from DHI.Generic.MikeZero import eumUnit, eumQuantity
from DHI.Generic.MikeZero.DFS import DfsFileFactory, DfsFactory
from DHI.Generic.MikeZero.DFS.dfsu import DfsuFile, DfsuFileType, DfsuBuilder, DfsuUtil
from DHI.Generic.MikeZero.DFS.mesh import MeshFile, MeshBuilder

from .dutil import get_item_info, get_valid_items_and_timesteps
from .dataset import Dataset
from .dotnet import (
    to_numpy,
    to_dotnet_float_array,
    to_dotnet_datetime,
    from_dotnet_datetime,
    asNumpyArray,
    to_dotnet_array,
    asnetarray_v2,
)
from .eum import ItemInfo, EUMType, EUMUnit
from .helpers import safe_length
from .spatial import Grid2D

class UnstructuredType(IntEnum):
    """
        -1: Mesh: 2D unstructured MIKE mesh
        0: Dfsu2D: 2D area series
        1: DfsuVerticalColumn: 1D vertical column
        2: DfsuVerticalProfileSigma: 2D vertical slice through a Dfsu3DSigma
        3: DfsuVerticalProfileSigmaZ: 2D vertical slice through a Dfsu3DSigmaZ
        4: Dfsu3DSigma: 3D file with sigma coordinates, i.e., a constant number of layers.
        5: Dfsu3DSigmaZ: 3D file with sigma and Z coordinates, i.e. a varying number of layers.
        """

    Mesh = -1
    Dfsu2D = 0
    DfsuVerticalColumn = 1
    DfsuVerticalProfileSigma = 2
    DfsuVerticalProfileSigmaZ = 3
    Dfsu3DSigma = 4
    Dfsu3DSigmaZ = 5


class _UnstructuredGeometry:
    # THIS CLASS KNOWS NOTHING ABOUT MIKE FILES!
    _type = None  # -1: mesh, 0: 2d-dfsu, 4:dfsu3dsigma, ...
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
    _element_table_dotnet = None

    _top_elems = None
    _n_layers_column = None
    _bot_elems = None
    _n_layers = None
    _n_sigma = None

    _geom2d = None
    _e2_e3_table = None
    _2d_ids = None
    _layer_ids = None

    _shapely_domain_obj = None
    _tree2d = None

    def __repr__(self):
        out = []
        out.append("Unstructured Geometry")
        if self.n_nodes:
            out.append(f"Number of nodes: {self.n_nodes}")
        if self.n_elements:
            out.append(f"Number of elements: {self.n_elements}")
        if self._n_layers:
            out.append(f"Number of layers: {self._n_layers}")
        if self._projstr:
            out.append(f"Projection: {self.projection_string}")
        return str.join("\n", out)

    @property
    def type_name(self):
        return self._type.name

    @property
    def n_nodes(self):
        """Number of nodes
        """
        return self._n_nodes

    @property
    def node_coordinates(self):
        """Coordinates (x,y,z) of all nodes
        """
        return self._nc

    @property
    def node_ids(self):
        return self._node_ids

    @property
    def n_elements(self):
        """Number of elements
        """
        return self._n_elements

    @property
    def element_ids(self):
        return self._element_ids

    @property
    def codes(self):
        """Node codes of all nodes
        """
        return self._codes

    @property
    def valid_codes(self):
        """Unique list of node codes
        """
        if self._valid_codes is None:
            self._valid_codes = list(set(self.codes))
        return self._valid_codes

    @property
    def boundary_codes(self):
        """provides a unique list of boundary codes
        """
        return [code for code in self.valid_codes if code > 0]

    @property
    def projection_string(self):
        return self._projstr

    @property
    def is_geo(self):
        """Are coordinates geographical (LONG/LAT)?
        """
        return self._projstr == "LONG/LAT"

    @property
    def is_local_coordinates(self):
        return self._projstr == "NON-UTM"

    @property
    def element_table(self):
        """Element to node connectivity
        """
        if (self._element_table is None) and (self._element_table_dotnet is not None):
            self._element_table = self._get_element_table_from_dotnet()
        return self._element_table

    @property
    def max_nodes_per_element(self):
        """The maximum number of nodes for an element
        """
        maxnodes = 0
        for local_nodes in self.element_table:
            n = len(local_nodes)
            if n > maxnodes:
                maxnodes = n
        return maxnodes

    @property
    def is_2d(self):
        """Type is either mesh or Dfsu2D (2 horizontal dimensions)
        """
        return self._type <= 0

    @property
    def is_tri_only(self):
        """Does the mesh consist of triangles only?
        """
        return self.max_nodes_per_element == 3 or self.max_nodes_per_element == 6

    @property
    def _shapely_domain2d(self):
        """
        """
        if self._shapely_domain_obj is None:
            self._shapely_domain_obj = self.to_shapely().buffer(0)
        return self._shapely_domain_obj

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
                print(
                    f"Selected code: {code} is not valid. Valid codes: {self.valid_codes}"
                )
                raise Exception
            return nc[self.codes == code]
        return nc

    def _get_element_table_from_dotnet(self):
        # Note: this can tak 10-20 seconds for large dfsu3d!
        elem_tbl = []
        for j in range(self.n_elements):
            elem_nodes = list(self._element_table_dotnet[j])
            elem_nodes = [nd - 1 for nd in elem_nodes]  # make 0-based
            elem_tbl.append(elem_nodes)
        return elem_tbl

    def _element_table_to_dotnet(self, elem_table=None):
        if elem_table is None:
            elem_table = self._element_table
        new_elem_table = []
        n_elements = len(elem_table)
        for j in range(n_elements):
            elem_nodes = elem_table[j]
            elem_nodes = [nd + 1 for nd in elem_nodes]  # make 1-based
            new_elem_table.append(elem_nodes)
        return asnetarray_v2(new_elem_table)

    def _set_nodes(
        self, node_coordinates, codes=None, node_ids=None, projection_string=None
    ):
        self._nc = np.asarray(node_coordinates)
        if codes is None:
            codes = np.zeros(len(node_coordinates), dtype=int)
        self._codes = np.asarray(codes)
        self._n_nodes = len(codes)
        if node_ids is None:
            node_ids = list(range(self._n_nodes))
        self._node_ids = np.asarray(node_ids)
        if projection_string is None:
            projection_string = "LONG/LAT"
        self._projstr = projection_string

    def _set_elements(self, element_table, element_ids=None, geometry_type=None):
        self._element_table = element_table
        self._n_elements = len(element_table)
        if element_ids is None:
            element_ids = list(range(self.n_elements))
        self._element_ids = np.asarray(element_ids)

        if geometry_type is None:
            # guess type
            if self.max_nodes_per_element < 5:
                geometry_type = UnstructuredType.Dfsu2D
            else:
                geometry_type = UnstructuredType.Dfsu3DSigma
        self._type = geometry_type

    def _reindex(self):
        new_node_ids = range(self.n_nodes)
        new_element_ids = range(self.n_elements)
        node_dict = dict(zip(self.node_ids, new_node_ids))
        for j in range(self.n_elements):
            elem_nodes = self._element_table[j]
            new_elem_nodes = []
            for idx in elem_nodes:
                new_elem_nodes.append(node_dict[idx])
            self._element_table[j] = new_elem_nodes

        self._node_ids = np.array(list(new_node_ids))
        self._element_ids = np.array(list(new_element_ids))

    def _get_element_table_for_elements(self, elements):
        return [self.element_table[j] for j in elements]

    def elements_to_geometry(self, elements, node_layers="all"):
        """export elements to new geometry

        Parameters
        ----------
        elements : list(int)
            list of element ids
        node_layers : str, optional
            for 3d files either 'top', 'bottom' layer nodes 
            or 'all' can be selected, by default 'all'

        Returns
        -------
        UnstructuredGeometry
            which can be used for further extraction or saved to file
        """
        elements = np.sort(elements)  # make sure elements are sorted!

        # extract information for selected elements
        node_ids, elem_tbl = self._get_nodes_and_table_for_elements(
            elements, node_layers=node_layers
        )
        node_coords = self.node_coordinates[node_ids]
        codes = self.codes[node_ids]

        # create new geometry
        geom = _UnstructuredGeometry()
        geom._set_nodes(
            node_coords,
            codes=codes,
            node_ids=node_ids,
            projection_string=self.projection_string,
        )
        geom._set_elements(elem_tbl, self.element_ids[elements])
        geom._reindex()

        geom._type = self._type  #
        if not self.is_2d:
            # original file was 3d

            layers_used = self.layer_ids[elements]
            unique_layer_ids = np.unique(layers_used)
            n_layers = len(unique_layer_ids)

            if (
                self._type == UnstructuredType.Dfsu3DSigma
                or self._type == UnstructuredType.Dfsu3DSigmaZ
            ) and n_layers == 1:
                # If source is 3d, but output only has 1 layer
                # then change type to 2d
                geom._type = UnstructuredType.Dfsu2D
                geom._n_layers = None
                if node_layers == "all":
                    print(
                        "Warning: Only 1 layer in new geometry (hence 2d), but you have kept both top and bottom nodes! Hint: use node_layers='top' or 'bottom'"
                    )
            else:
                geom._type = self._type
                geom._n_layers = n_layers
                lowest_sigma = self.n_layers - self.n_sigma_layers + 1
                geom._n_sigma = sum(unique_layer_ids >= lowest_sigma)

                # If source is sigma-z but output only has sigma layers
                # then change type accordingly
                if (
                    self._type == UnstructuredType.DfsuVerticalProfileSigmaZ
                    or self._type == UnstructuredType.Dfsu3DSigmaZ
                ) and n_layers == geom._n_sigma:
                    geom._type = UnstructuredType(self._type.value - 1)

                geom._top_elems = geom._get_top_elements_from_coordinates()

        return geom

    def _get_top_elements_from_coordinates(self, ec=None):
        """Get list of top element ids based on element coordinates        
        """
        if ec is None:
            ec = self.element_coordinates

        d_eps = 1e-4
        top_elems = []
        x_old = ec[0, 0]
        y_old = ec[0, 1]
        for j in range(1, len(ec)):
            d2 = (ec[j, 0] - x_old) ** 2 + (ec[j, 1] - y_old) ** 2
            # print(d2)
            if d2 > d_eps:
                # this is a new x,y point
                # then the previous element must be a top element
                top_elems.append(j - 1)
            x_old = ec[j, 0]
            y_old = ec[j, 1]
        return np.array(top_elems)

    def to_2d_geometry(self):
        """extract 2d geometry from 3d geometry

        Returns
        -------
        UnstructuredGeometry
            2d geometry (bottom nodes)
        """
        if self._n_layers is None:
            print("Object has no layers: cannot export to_2d_geometry")
            return None

        # extract information for selected elements
        elem_ids = self.bottom_elements
        node_ids, elem_tbl = self._get_nodes_and_table_for_elements(
            elem_ids, node_layers="bottom"
        )
        node_coords = self.node_coordinates[node_ids]
        codes = self.codes[node_ids]

        # create new geometry
        geom = _UnstructuredGeometry()
        geom._set_nodes(
            node_coords,
            codes=codes,
            node_ids=node_ids,
            projection_string=self.projection_string,
        )
        geom._set_elements(elem_tbl, self.element_ids[elem_ids])

        geom._type = UnstructuredType.Mesh

        geom._reindex()

        return geom

    def _get_nodes_and_table_for_elements(self, elements, node_layers="all"):
        """list of nodes and element table for a list of elements

        Parameters
        ----------
        elements : np.array(int)
            array of element ids
        node_layers : str, optional
            for 3D files 'all', 'bottom' or 'top' nodes 
            of each element, by default 'all'

        Returns
        -------
        np.array(int)
            array of node ids (unique)
        list(list(int))
            element table with a list of nodes for each element
        """
        nodes = []
        elem_tbl = []
        if (node_layers is None) or (node_layers == "all") or self.is_2d:
            for j in elements:
                elem_nodes = self.element_table[j]
                elem_tbl.append(elem_nodes)
                for node in elem_nodes:
                    nodes.append(node)
        else:
            # 3D file
            if (node_layers != "bottom") and (node_layers != "top"):
                raise Exception("node_layers must be either all, bottom or top")
            for j in elements:
                elem_nodes = self.element_table[j]
                nn = len(elem_nodes)
                halfn = int(nn / 2)
                if node_layers == "bottom":
                    elem_nodes = elem_nodes[:halfn]
                if node_layers == "top":
                    elem_nodes = elem_nodes[halfn:]
                elem_tbl.append(elem_nodes)
                for node in elem_nodes:
                    nodes.append(node)

        return np.unique(nodes), elem_tbl

    @property
    def element_coordinates(self):
        """Center coordinates of each element
        """
        if self._ec is None:
            self._ec = self._get_element_coords()
        return self._ec

    def _get_element_coords(self):
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
                idx[i] = nodes[i]  # - 1

            xcoords[:nnodes, j] = self._nc[idx[:nnodes], 0]
            ycoords[:nnodes, j] = self._nc[idx[:nnodes], 1]
            zcoords[:nnodes, j] = self._nc[idx[:nnodes], 2]

        ec[:, 0] = np.sum(xcoords, axis=0) / nnodes_per_elem
        ec[:, 1] = np.sum(ycoords, axis=0) / nnodes_per_elem
        ec[:, 2] = np.sum(zcoords, axis=0) / nnodes_per_elem

        self._ec = ec
        return ec

    def contains(self, points):
        """test if a list of points are contained by mesh

        Parameters
        ----------
        points : array-like n-by-2
            x,y-coordinates of n points to be tested

        Returns
        -------
        bool array
            True for points inside, False otherwise
        """
        import matplotlib.path as mp
        try:
            domain = self._shapely_domain2d
        except:
            warnings.warn('Could not determine if domain contains points. Failed to convert to_shapely()')
            return None
        
        cnts = mp.Path(domain.exterior).contains_points(points)
        for interj in domain.interiors:
            in_hole = mp.Path(interj).contains_points(points) 
            cnts = np.logical_and(cnts, ~in_hole)
        return cnts


    def _create_tree2d(self):
        xy = self.geometry2d.element_coordinates[:,:2]
        self._tree2d = cKDTree(xy)

    def _find_n_nearest_2d_elements(self, x, y=None, n_points=1):        
        if self._tree2d is None:
            self._create_tree2d()

        if y is None: 
            p = x
        else:
            p = np.array((x, y))
        d, elem_id = self._tree2d.query(p, k=n_points)
        return elem_id, d

    def get_overset_grid(self, dxdy=None, shape=None):
        """get a 2d grid that covers the domain by specifying spacing or shape

        Parameters
        ----------
        dxdy : float or (float, float), optional
            grid resolution in x- and y-direction
        shape : (int, int), optional
            tuple with nx and ny describing number of points in each direction
            one of them can be None, in which case the value will be inferred

        Returns
        -------
        <mikeio.Grid2D>
            2d grid
        """
        coords = self.geometry2d.node_coordinates   # node_ or element_
        small = 1e-10        
        x0 = coords[:,0].min() + small
        y0 = coords[:,1].min() + small
        x1 = coords[:,0].max() - small
        y1 = coords[:,1].max() - small
        bbox = [x0, y0, x1, y1]
        return Grid2D(bbox=bbox, dxdy=dxdy, shape=shape)
        
    def get_2d_interpolant(self, xy, n_nearest:int=1, extrapolate=False):
        """IDW interpolant for list of coordinates

        Parameters
        ----------
        xy : array-like 
            x,y coordinates of new points 
        n_nearest : int, optional
            [description], by default 1
        extrapolate : bool, optional
            allow , by default False

        Returns
        -------
        (np.array, np.array)
            element ids and weights 
        """
        elem_ids, distances = self._find_n_nearest_2d_elements(xy, n_points=n_nearest)

        if n_nearest == 1:
            weights = np.ones(distances.shape)
            if not extrapolate: 
                weights[~self.contains(xy)] = np.nan
        elif n_nearest > 1:
            weights = np.zeros(distances.shape)
            p = 1   # inverse distance order 
            
            match = distances[:,0] < 1e-8
            weights[match,0] = 1
            
            weights[~match,:] = 1 / distances[~match,:]**p
            denom = weights[~match,:].sum(axis=1).reshape(-1,1)*np.ones((1,n_nearest))
            weights[~match,:] = weights[~match,:] / denom
            if not extrapolate: 
                weights[~self.contains(xy),:] = np.nan
        else:
            ValueError('n_nearest must be at least 1')

        return elem_ids, weights

    # def gridded_2d_interpolant(self, x0, dx, nx:int, y0, dy, ny:int, n_nearest:int=1):
    #     x1 = x0 + dx*(nx-1)
    #     x = np.linspace(x0, x1, nx)
    #     y1 = y0 + dy*(ny-1)
    #     y = np.linspace(y0, y1, ny)
    #     xv, yv = np.meshgrid(x, y)
    #     xy = np.empty((nx*ny, 2))
    #     xy[:,0] = xv.reshape(-1)
    #     xy[:,1] = yv.reshape(-1)
    #     return self.get_2d_interpolant(xy, n_nearest)

    def interp2d(self, data, interpolant2d):
        
        if len(data) == self.n_elements:
            return self._interp_itemstep(data, interpolant2d)

        idat = []
        ni = len(interpolant2d[0])
        for datitem in data:
            nt, ne = datitem.shape
            idatitem = np.empty(shape=(nt,ni)) 
            for step in range(nt):
                idatitem[step,:] = self._interp_itemstep(datitem[step,:], interpolant2d)
            idat.append(idatitem)
        return idat

    def _interp_itemstep(self, data, interpolant2d):
        ix = interpolant2d[0]
        w = interpolant2d[1]
        ni = len(ix)
        idat = np.empty(ni)
        for j in range(ni):
            idat[j] = np.dot(data[ix[j]], w[j])
        return idat

    def _find_n_nearest_elements(self, x, y, z=None, n=1, layer=None):
        """Find n nearest elements (for each of the points given) 

        Parameters
        ----------

        x: float or list(float)
            X coordinate(s) (easting or longitude)
        y: float or list(float)
            Y coordinate(s) (northing or latitude)
        z: float or list(float), optional
            Z coordinate(s)  (vertical coordinate, positive upwards)
            If not provided for a 3d file, the surface element is returned
        layer: int, optional
            Search in a specific layer only (3D files only)

        Returns
        -------
        np.array
            element ids of nearest element(s)            
        """
        ec = self.element_coordinates

        if self.is_2d:
            poi = np.array([x, y])

            d = ((ec[:, 0:2] - poi) ** 2).sum(axis=1)
            idx = d.argsort()[0:n]
        else:
            poi = np.array([x, y])

            ec = self.geometry2d.element_coordinates
            d2d = ((ec[:, 0:2] - poi) ** 2).sum(axis=1)
            elem2d = d2d.argsort()[0:n]  # n nearest 2d elements

            if layer is None:
                # TODO: loop over 2d elements, to get n lateral 3d neighbors
                elem3d = self.e2_e3_table[elem2d[0]]
                zc = self.element_coordinates[elem3d, 2]

                if z is None:
                    z = 0  # should we rarther return whole column?
                d3d = np.abs(z - zc)
                idx = elem3d[d3d.argsort()[0]]
            else:
                # 3d elements for n nearest 2d elements
                elem3d = self.e2_e3_table[elem2d]
                elem3d = np.concatenate(elem3d, axis=0)
                layer_ids = self.layer_ids[elem3d]
                idx = elem3d[layer_ids == layer]  # return at most n ids

        if n == 1 and (not np.isscalar(idx)):
            idx = idx[0]
        return idx

    def find_nearest_element(self, x, y, z=None, layer=None):
        """Find index of nearest element (optionally for a list)

        Parameters
        ----------

        x: float or list(float)
            X coordinate(s) (easting or longitude)
        y: float or list(float)
            Y coordinate(s) (northing or latitude)
        z: float or list(float), optional
            Z coordinate(s)  (vertical coordinate, positive upwards)
            If not provided for a 3d file, the surface element is returned
        layer: int, optional
            Search in a specific layer only (3D files only)

        Returns
        -------
        np.array
            element ids of nearest element(s)
        """
        if np.isscalar(x):
            return self._find_n_nearest_elements(x, y, z, n=1, layer=layer)
        else:
            nx = len(x)
            ny = len(y)
            if nx != ny:
                print(f"x and y must have same length")
                raise Exception
            idx = np.zeros(nx, dtype=int)
            if z is None:
                for j in range(nx):
                    idx[j] = self._find_n_nearest_elements(
                        x[j], y[j], z=None, n=1, layer=layer
                    )
            else:
                nz = len(z)
                if nx != nz:
                    print(f"z must have same length as x and y")
                for j in range(nx):
                    idx[j] = self._find_n_nearest_elements(
                        x[j], y[j], z[j], n=1, layer=layer
                    )
        return idx

    # def _find_nearest_2d_element(self, x, y):
    #     if self.is_2d:
    #         return self.find_nearest_element(x, y)
    #     else:
    #         geom2d = self.geometry2d
    #         return geom2d.find_nearest_element(x, y)

    # def _get_profile_from_2d_element(self, elem2d):
    #     if self.is_2d:
    #         raise Exception("Object is 2d. Cannot get_profile_from_2d_element")
    #     else:
    #         return self.e2_e3_table[elem2d]

    def find_nearest_profile_elements(self, x, y):
        """Find 3d elements of profile nearest to (x,y) coordinates

        Parameters
        ----------
        x : float
            x-coordinate of point
        y : float
            y-coordinate of point

        Returns
        -------
        np.array(int)
            element ids of vertical profile
        """
        if self.is_2d:
            raise Exception("Object is 2d. Cannot get_nearest_profile")
        else:
            elem2d = self.geometry2d.find_nearest_element(x, y)
            elem3d = self.e2_e3_table[elem2d]
            return elem3d

    def get_element_area(self):
        """Calculate the horizontal area of each element.

        Returns:
        np.array(float)
            areas in m2
        """
        n_elements = self.n_elements

        # Node coordinates
        xn = self.node_coordinates[:, 0]
        yn = self.node_coordinates[:, 1]

        area = np.empty(n_elements)
        xcoords = np.empty(8)
        ycoords = np.empty(8)

        for j in range(n_elements):
            nodes = self.element_table[j]
            n_nodes = len(nodes)

            for i in range(n_nodes):
                nidx = nodes[i]
                xcoords[i] = xn[nidx]
                ycoords[i] = yn[nidx]

            # ab : edge vector corner a to b
            abx = xcoords[1] - xcoords[0]
            aby = ycoords[1] - ycoords[0]

            # ac : edge vector corner a to c
            acx = xcoords[2] - xcoords[0]
            acy = ycoords[2] - ycoords[0]

            isquad = False
            if n_nodes > 3:
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
                Ye = np.sum(ycoords[:n_nodes]) / n_nodes
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

    # 3D dfsu stuff
    @property
    def geometry2d(self):
        """The 2d geometry for a 3d object
        """
        if self._n_layers is None:
            #print("Object has no layers: cannot return geometry2d")
            return self
        if self._geom2d is None:
            self._geom2d = self.to_2d_geometry()
        return self._geom2d

    @property
    def e2_e3_table(self):
        """The 2d-to-3d element connectivity table for a 3d object
        """
        if self._n_layers is None:
            print("Object has no layers: cannot return e2_e3_table")
            return None
        if self._e2_e3_table is None:
            res = self._get_2d_to_3d_association()
            self._e2_e3_table = res[0]
            self._2d_ids = res[1]
            self._layer_ids = res[2]
        return self._e2_e3_table

    @property
    def elem2d_ids(self):
        """The associated 2d element id for each 3d element
        """
        if self._n_layers is None:
            print("Object has no layers: cannot return elem2d_ids")
            return None
        if self._2d_ids is None:
            res = self._get_2d_to_3d_association()
            self._e2_e3_table = res[0]
            self._2d_ids = res[1]
            self._layer_ids = res[2]
        return self._2d_ids

    @property
    def layer_ids(self):
        """The layer number for each 3d element
        """
        if self._n_layers is None:
            print("Object has no layers: cannot return layer_ids")
            return None
        if self._layer_ids is None:
            res = self._get_2d_to_3d_association()
            self._e2_e3_table = res[0]
            self._2d_ids = res[1]
            self._layer_ids = res[2]
        return self._layer_ids

    @property
    def n_layers(self):
        """Maximum number of layers
        """
        return self._n_layers

    @property
    def n_sigma_layers(self):
        """Number of sigma layers
        """
        return self._n_sigma

    @property
    def n_z_layers(self):
        """Maximum number of z-layers
        """
        if self._n_layers is None:
            return None
        return self._n_layers - self._n_sigma

    @property
    def top_elements(self):
        """List of 3d element ids of surface layer
        """
        if self._n_layers is None:
            print("Object has no layers: cannot find top_elements")
            return None
        elif (self._top_elems is None) and (self._source is not None):
            # note: if subset of elements is selected then this cannot be done!
            self._top_elems = np.array(DfsuUtil.FindTopLayerElements(self._source))
        return self._top_elems

    @property
    def n_layers_per_column(self):
        """List of number of layers for each column
        """
        if self._n_layers is None:
            print("Object has no layers: cannot find n_layers_per_column")
            return None
        elif self._n_layers_column is None:
            top_elems = self.top_elements
            n = len(top_elems)
            tmp = top_elems.copy()
            tmp[0] = -1
            tmp[1:n] = top_elems[0 : (n - 1)]
            self._n_layers_column = top_elems - tmp
        return self._n_layers_column

    @property
    def bottom_elements(self):
        """List of 3d element ids of bottom layer
        """
        if self._n_layers is None:
            print("Object has no layers: cannot find bottom_elements")
            return None
        elif self._bot_elems is None:
            self._bot_elems = self.top_elements - self.n_layers_per_column + 1
        return self._bot_elems

    def get_layer_elements(self, layer):
        """3d element ids for one (or more) specific layer(s)

        Parameters
        ----------
        layer : int or list(int)
            layer between 1 (bottom) and n_layers (top) 
            (can also be negative counting from 0 at the top layer)

        Returns
        -------
        np.array(int)
            element ids
        """
        if not np.isscalar(layer):
            elem_ids = []
            for nn in layer:
                elem_ids.append(self.get_layer_elements(nn))
            elem_ids = np.concatenate(elem_ids, axis=0)
            return np.sort(elem_ids)

        n_lay = self.n_layers
        if n_lay is None:
            print("Object has no layers: cannot get_layer_elements")
            return None

        if layer < (-n_lay + 1) or layer > n_lay:
            raise Exception(
                f"Layer {layer} not allowed; must be between -{n_lay-1} and {n_lay}"
            )

        if layer <= 0:
            layer = layer + n_lay

        return self.element_ids[self.layer_ids == layer]

    def _get_2d_to_3d_association(self):
        e2_to_e3 = (
            []
        )  # for each 2d element: the corresponding 3d element ids from bot to top
        index2d = []  # for each 3d element: the associated 2d element id
        layerid = []  # for each 3d element: the associated layer number
        n2d = len(self.top_elements)
        topid = self.top_elements
        botid = self.bottom_elements
        global_layer_ids = np.arange(1, self.n_layers + 1)  # layer_ids = 1, 2, 3...
        for j in range(n2d):
            col = list(range(botid[j], topid[j] + 1))

            e2_to_e3.append(col)
            for jj in col:
                index2d.append(j)

            n_local_layers = len(col)
            local_layers = global_layer_ids[-n_local_layers:]
            for ll in local_layers:
                layerid.append(ll)

        e2_to_e3 = np.array(e2_to_e3)
        index2d = np.array(index2d)
        layerid = np.array(layerid)
        return e2_to_e3, index2d, layerid

    def _to_polygons(self, geometry=None):
        """generate matplotlib polygons from element table for plotting

        Returns
        -------
        list(matplotlib.patches.Polygon)
            list of polygons for plotting
        """
        if geometry is None:
            geometry = self
        from matplotlib.patches import Polygon

        polygons = []

        for j in range(geometry.n_elements):
            nodes = geometry.element_table[j]
            pcoords = np.empty([len(nodes), 2])
            for i in range(len(nodes)):
                nidx = nodes[i]
                pcoords[i, :] = geometry.node_coordinates[nidx, 0:2]

            polygon = Polygon(pcoords, True)
            polygons.append(polygon)
        return polygons

    def to_shapely(self):
        """Export mesh as shapely MultiPolygon

        Returns
        -------
        shapely.geometry.MultiPolygon
            polygons with mesh elements
        """
        from shapely.geometry import Polygon, MultiPolygon

        polygons = []
        for j in range(self.n_elements):
            nodes = self.element_table[j]
            pcoords = np.empty([len(nodes), 2])
            for i in range(len(nodes)):
                nidx = nodes[i]
                pcoords[i, :] = self.node_coordinates[nidx, 0:2]
            polygon = Polygon(pcoords)
            polygons.append(polygon)
        mp = MultiPolygon(polygons)

        return mp

    def get_node_centered_data(self, data, extrapolate=True):
        """convert cell-centered data to node-centered by pseudo-laplacian method

        Parameters
        ----------
        data : np.array(float)
            cell-centered data 
        extrapolate : bool, optional
            allow the method to extrapolate, default:True

        Returns
        -------
        np.array(float)
            node-centered data 
        """
        nc = self.node_coordinates
        elem_table, ec, data = self._create_tri_only_element_table(data)

        node_cellID = [
            list(np.argwhere(elem_table == i)[:, 0])
            for i in np.unique(elem_table.reshape(-1,))
        ]
        node_centered_data = np.zeros(shape=nc.shape[0])
        for n, item in enumerate(node_cellID):
            I = ec[item][:, :2] - nc[n][:2]
            I2 = (I ** 2).sum(axis=0)
            Ixy = (I[:, 0] * I[:, 1]).sum(axis=0)
            lamb = I2[0] * I2[1] - Ixy ** 2
            omega = np.zeros(1)
            if lamb > 1e-10 * (I2[0] * I2[1]):
                # Standard case - Pseudo
                lambda_x = (Ixy * I[:, 1] - I2[1] * I[:, 0]) / lamb
                lambda_y = (Ixy * I[:, 0] - I2[0] * I[:, 1]) / lamb
                omega = 1.0 + lambda_x * I[:, 0] + lambda_y * I[:, 1]
                if not extrapolate:
                    omega[np.where(omega > 2)] = 2
                    omega[np.where(omega < 0)] = 0
            if omega.sum() > 0:
                node_centered_data[n] = np.sum(omega * data[item]) / np.sum(omega)
            else:
                # We did not succeed using pseudo laplace procedure, use inverse distance instead
                InvDis = [
                    1 / np.hypot(case[0], case[1])
                    for case in ec[item][:, :2] - nc[n][:2]
                ]
                node_centered_data[n] = np.sum(InvDis * data[item]) / np.sum(InvDis)

        return node_centered_data

    def plot(
        self,
        z=None,
        elements=None,
        plot_type="patch",
        title=None,
        label=None,
        cmap=None,
        vmin=None,
        vmax=None,
        levels=10,
        n_refinements=0,
        show_mesh=True,
        show_outline=True,
        figsize=None,
        ax=None,
    ):
        """
        Plot unstructured data and/or mesh, mesh outline  

        Parameters
        ----------
        z: np.array, optional
            value for each element to plot, default bathymetry
        elements: list(int), optional
            list of element ids to be plotted
        plot_type: str, optional 
            type of plot: 'patch' (default), 'mesh_only', 'shaded', 
            'contour', 'contourf' or 'outline_only' 
        title: str, optional
            axes title 
        label: str, optional
            colorbar label (or title if contour plot)
        cmap: matplotlib.cm.cmap, optional
            colormap, default viridis            
        vmin: real, optional 
            lower bound of values to be shown on plot, default:None 
        vmax: real, optional 
            upper bound of values to be shown on plot, default:None 
        levels: int, list(float), optional
            for contour plots: how many levels, default:10
            or a list of discrete levels e.g. [3.0, 4.5, 6.0]
        show_mesh: bool, optional
            should the mesh be shown on the plot? default=True
        show_outline: bool, optional
            should domain outline be shown on the plot? default=True
        n_refinements: int, optional
            for 'shaded' and 'contour' plots (and if show_mesh=False) 
            do this number of mesh refinements for smoother plotting  
        figsize: (float, float), optional
            specify size of figure
        ax: matplotlib.axes, optional
            Adding to existing axis, instead of creating new fig

        Returns
        -------
        <matplotlib.axes>          
        """

        import matplotlib.cm as cm
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        from matplotlib.collections import PatchCollection
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        mesh_col = "0.95"
        mesh_col_dark = "0.6"

        if plot_type is None:
            plot_type = "outline_only"

        plot_data = True
        if plot_type == "mesh_only" or plot_type == "outline_only":
            plot_data = False

        if cmap is None:
            cmap = cm.viridis

        if elements is None:
            if self.is_2d:
                geometry = self
            else:
                geometry = self.geometry2d
        else:
            # spatial subset
            if self.is_2d:
                geometry = self.elements_to_geometry(elements)
            else:
                geometry = self.elements_to_geometry(elements, node_layers="bottom")

        nc = geometry.node_coordinates
        ec = geometry.element_coordinates
        ne = ec.shape[0]

        is_bathy = False
        if z is None:
            is_bathy = True
            if plot_data:
                z = ec[:, 2]
                if label is None:
                    label = "Bathymetry (m)"
        else:
            if len(z) != ne:
                raise Exception(
                    f"Length of z ({len(z)}) does not match geometry ({ne})"
                )
            if label is None:
                label = ""
            if not plot_data:
                print(f"Cannot plot data in {plot_type} plot!")

        if plot_data and vmin is None:
            vmin = np.nanmin(z)
        if plot_data and vmax is None:
            vmax = np.nanmax(z)

        # set levels
        if "contour" in plot_type:
            if levels is None:
                levels = 10
            if np.isscalar(levels):
                n_levels = levels
                levels = np.linspace(vmin, vmax, n_levels)
            else:
                n_levels = len(levels)
                vmin = min(levels)
                vmax = max(levels)

        # plot in existing or new axes?
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # set aspect ratio
        if geometry.is_geo:
            mean_lat = np.mean(nc[:, 1])
            ax.set_aspect(1.0 / np.cos(np.pi * mean_lat / 180))
        else:
            ax.set_aspect("equal")

        # set plot limits
        xmin, xmax = nc[:, 0].min(), nc[:, 0].max()
        ymin, ymax = nc[:, 1].min(), nc[:, 1].max()

        # scale height of colorbar
        cbar_frac = 0.046 * nc[:, 1].ptp() / nc[:, 0].ptp()

        if plot_type == "outline_only":
            fig_obj = None

        elif plot_type == "mesh_only":
            if show_mesh == False:
                print("Not possible to use show_mesh=False on a mesh_only plot!")
            patches = geometry._to_polygons()
            fig_obj = PatchCollection(
                patches, edgecolor=mesh_col_dark, facecolor="none", linewidths=0.3
            )
            ax.add_collection(fig_obj)

        elif plot_type == "patch" or plot_type == "box":
            patches = geometry._to_polygons()
            # do plot as patches (like MZ "box contour")
            # with (constant) element center values
            if show_mesh:
                fig_obj = PatchCollection(
                    patches, cmap=cmap, edgecolor=mesh_col, linewidths=0.4
                )
            else:
                fig_obj = PatchCollection(
                    patches, cmap=cmap, edgecolor="face", alpha=None, linewidths=None
                )

            fig_obj.set_array(z)
            fig_obj.set_clim(vmin, vmax)
            ax.add_collection(fig_obj)

            cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
            plt.colorbar(fig_obj, label=label, cax=cax)

        else:
            # do node-based triangular plot
            import matplotlib.tri as tri

            mesh_linewidth = 0.0
            if show_mesh and geometry.is_tri_only:
                mesh_linewidth = 0.4
                if n_refinements > 0:
                    n_refinements = 0
                    print("Warning: mesh refinement is not possible if plot_mesh=True")

            elem_table, ec, z = self._create_tri_only_element_table(
                data=z, geometry=geometry
            )
            triang = tri.Triangulation(nc[:, 0], nc[:, 1], elem_table)

            zn = geometry.get_node_centered_data(z)

            if n_refinements > 0:
                # TODO: refinements doesn't seem to work for 3d files?
                refiner = tri.UniformTriRefiner(triang)
                triang, zn = refiner.refine_field(zn, subdiv=n_refinements)

            if plot_type == "shaded" or plot_type == "smooth":
                ax.triplot(triang, lw=mesh_linewidth, color=mesh_col)
                fig_obj = ax.tripcolor(
                    triang,
                    zn,
                    edgecolors="face",
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                    linewidths=0.3,
                    shading="gouraud",
                )

                cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
                plt.colorbar(fig_obj, label=label, cax=cax)

            elif plot_type == "contour" or plot_type == "contour_lines":
                ax.triplot(triang, lw=mesh_linewidth, color=mesh_col_dark)
                fig_obj = ax.tricontour(
                    triang, zn, levels=levels, linewidths=[1.2], cmap=cmap
                )
                ax.clabel(fig_obj, fmt="%1.2f", inline=1, fontsize=9)
                if len(label) > 0:
                    ax.set_title(label)

            elif plot_type == "contourf" or plot_type == "contour_filled":
                ax.triplot(triang, lw=mesh_linewidth, color=mesh_col)
                vbuf = 0.01 * (vmax - vmin) / n_levels
                zn = np.clip(zn, vmin + vbuf, vmax - vbuf)  # avoid white outside limits
                fig_obj = ax.tricontourf(triang, zn, levels=levels, cmap=cmap)

                # colorbar
                cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
                plt.colorbar(fig_obj, label=label, cax=cax)

            else:
                if (plot_type is not None) and plot_type != "outline_only":
                    raise Exception(f"plot_type {plot_type} unknown!")

            if show_mesh and (not geometry.is_tri_only):
                # if mesh is not tri only, we need to add it manually on top
                patches = geometry._to_polygons()
                mesh_linewidth = 0.4
                if plot_type == "contour":
                    mesh_col = mesh_col_dark
                p = PatchCollection(
                    patches,
                    edgecolor=mesh_col,
                    facecolor="none",
                    linewidths=mesh_linewidth,
                )
                ax.add_collection(p)

        if show_outline:
            try:
                domain = self._shapely_domain2d
            except:
                warnings.warn("Could not plot outline. Failed to convert to_shapely()")
            try:
                if domain:
                    out_col = "0.4"
                    ax.plot(*domain.exterior.xy, color=out_col, linewidth=1.2)
                    xd, yd = domain.exterior.xy[0], domain.exterior.xy[1]
                    xmin, xmax = min(xmin, np.min(xd)), max(xmax, np.max(xd))
                    ymin, ymax = min(ymin, np.min(yd)), max(ymax, np.max(yd))
                    for j in range(len(domain.interiors)):
                        interj = domain.interiors[j]
                        ax.plot(*interj.xy, color=out_col, linewidth=1.2)
            except:
                warnings.warn("Could not plot outline")

        # set plot limits
        xybuf = 6e-3 * (xmax - xmin)
        ax.set_xlim(xmin - xybuf, xmax + xybuf)
        ax.set_ylim(ymin - xybuf, ymax + xybuf)

        if title is not None:
            ax.set_title(title)

        return ax

    def _create_tri_only_element_table(self, data=None, geometry=None):
        """Convert quad/tri mesh to pure tri-mesh
        """
        if geometry is None:
            geometry = self

        ec = geometry.element_coordinates
        if geometry.is_tri_only:
            return np.asarray(geometry.element_table), ec, data

        if data is None:
            data = []

        elem_table = [
            list(geometry.element_table[i]) for i in range(geometry.n_elements)
        ]
        tmp_elmnt_nodes = elem_table.copy()
        for el, item in enumerate(tmp_elmnt_nodes):
            if len(item) == 4:
                elem_table.pop(el)  # remove quad element

                # insert two new tri elements in table
                elem_table.insert(el, item[:3])
                tri2_nodes = [item[i] for i in [2, 3, 0]]
                elem_table.append(tri2_nodes)

                # new center coordinates for new tri-elements
                ec[el] = geometry.node_coordinates[item[:3]].mean(axis=1)
                tri2_ec = geometry.node_coordinates[tri2_nodes].mean(axis=1)
                ec = np.append(ec, tri2_ec.reshape(1, -1), axis=0)

                # use same data in two new tri elements
                data = np.append(data, data[el])

        return np.asarray(elem_table), ec, data


class _UnstructuredFile(_UnstructuredGeometry):
    """
    _UnstructuredFile based on _UnstructuredGeometry and base class for Mesh and Dfsu
    knows dotnet file, items and timesteps and reads file header 
    """

    _filename = None
    _source = None
    _deletevalue = None

    _n_timesteps = None
    _start_time = None
    _timestep_in_seconds = None

    _n_items = None
    _items = None
    _dtype = np.float64

    def __repr__(self):
        out = []
        if self._type is not None:
            out.append(self.type_name)
        out.append(f"Number of elements: {self.n_elements}")
        out.append(f"Number of nodes: {self.n_nodes}")
        if self._projstr:
            out.append(f"Projection: {self.projection_string}")
        if not self.is_2d:
            out.append(f"Number of sigma layers: {self.n_sigma_layers}")
        if (
            self._type == UnstructuredType.DfsuVerticalProfileSigmaZ
            or self._type == UnstructuredType.Dfsu3DSigmaZ
        ):
            out.append(f"Max number of z layers: {self.n_layers - self.n_sigma_layers}")
        if self._n_items is not None:
            if self._n_items < 10:
                out.append("Items:")
                for i, item in enumerate(self.items):
                    out.append(f"  {i}:  {item}")
            else:
                out.append(f"Number of items: {self._n_items}")
        if self._n_timesteps is not None:
            if self._n_timesteps == 1:
                out.append(f"Time: time-invariant file (1 step) at {self._start_time}")
            else:
                out.append(
                    f"Time: {self._n_timesteps} steps with dt={self._timestep_in_seconds}s"
                )
                out.append(f"      {self._start_time} -- {self.end_time}")
        return str.join("\n", out)

    def __init__(self):
        super().__init__()

    def _read_header(self, filename):
        if not os.path.isfile(filename):
            raise Exception(f"file {filename} does not exist!")

        _, ext = os.path.splitext(filename)

        if ext == ".mesh":
            self._read_mesh_header(filename)

        elif ext == ".dfsu":
            self._read_dfsu_header(filename)
        else:
            raise Exception(f"Filetype {ext} not supported (mesh,dfsu)")

    def _read_mesh_header(self, filename):
        """
        Read header of mesh file and set object properties
        """
        msh = MeshFile.ReadMesh(filename)
        self._source = msh
        self._projstr = msh.ProjectionString
        self._type = UnstructuredType.Mesh

        # geometry
        self._set_nodes_from_source(msh)
        self._set_elements_from_source(msh)

    def _read_dfsu_header(self, filename):
        """
        Read header of dfsu file and set object properties
        """
        dfs = DfsuFile.Open(filename)
        self._source = dfs
        self._projstr = dfs.Projection.WKTString
        self._type = UnstructuredType(dfs.DfsuFileType)
        self._deletevalue = dfs.DeleteValueFloat

        # geometry
        self._set_nodes_from_source(dfs)
        self._set_elements_from_source(dfs)

        if not self.is_2d:
            self._n_layers = dfs.NumberOfLayers
            self._n_sigma = dfs.NumberOfSigmaLayers

        # items
        self._n_items = safe_length(dfs.ItemInfo)
        self._items = get_item_info(dfs, list(range(self._n_items)))

        # time
        self._start_time = from_dotnet_datetime(dfs.StartDateTime)
        self._n_timesteps = dfs.NumberOfTimeSteps
        self._timestep_in_seconds = dfs.TimeStepInSeconds

        dfs.Close()

    def _set_nodes_from_source(self, source):
        xn = asNumpyArray(source.X)
        yn = asNumpyArray(source.Y)
        zn = asNumpyArray(source.Z)
        self._nc = np.column_stack([xn, yn, zn])
        self._codes = np.array(list(source.Code))
        self._n_nodes = source.NumberOfNodes
        self._node_ids = np.array(list(source.NodeIds)) - 1

    def _set_elements_from_source(self, source):
        self._n_elements = source.NumberOfElements
        self._element_table_dotnet = source.ElementTable
        self._element_table = None  # do later if needed
        self._element_ids = np.array(list(source.ElementIds)) - 1


class Dfsu(_UnstructuredFile):
    def __init__(self, filename, dtype=np.float64):
        """
        Create a Dfsu object

        Parameters
        ---------
        filename: str
            dfsu or mesh filename
        dtype: np.dtype, optional
            default np.float64, valid options are np.float32, np.float64
        """
        if dtype not in [np.float32, np.float64]:
            raise ValueError("Invalid data type. Choose np.float32 or np.float64")

        super().__init__()
        self._filename = filename
        self._read_header(filename)
        self._dtype = dtype

    @property
    def element_coordinates(self):
        # faster way of getting element coordinates than base class implementation
        if self._ec is None:
            self._ec = self._get_element_coords_from_source()
        return self._ec

    def _get_element_coords_from_source(self):
        xc = np.zeros(self.n_elements)
        yc = np.zeros(self.n_elements)
        zc = np.zeros(self.n_elements)
        _, xc2, yc2, zc2 = DfsuUtil.CalculateElementCenterCoordinates(
            self._source, to_dotnet_array(xc), to_dotnet_array(yc), to_dotnet_array(zc),
        )
        ec = np.column_stack([asNumpyArray(xc2), asNumpyArray(yc2), asNumpyArray(zc2)])
        return ec

    @property
    def deletevalue(self):
        """File delete value
        """
        return self._deletevalue

    @property
    def n_items(self):
        """Number of items
        """
        return self._n_items

    @property
    def items(self):
        """List of items
        """
        return self._items

    @property
    def start_time(self):
        """File start time
        """
        return self._start_time

    @property
    def n_timesteps(self):
        """Number of time steps
        """
        return self._n_timesteps

    @property
    def timestep(self):
        """Time step size in seconds
        """
        return self._timestep_in_seconds

    # @timestep.setter
    # def timestep(self, value):
    #     if value <= 0:
    #         print(f'timestep must be positive scalar!')
    #     else:
    #         self._timestep_in_seconds = value

    @property
    def end_time(self):
        """File end time
        """
        return self.start_time + timedelta(
            seconds=((self.n_timesteps - 1) * self.timestep)
        )

    def read(self, items=None, time_steps=None, elements=None):
        """
        Read data from a dfsu file

        Parameters
        ---------
        filename: str
            dfsu filename
        items: list[int] or list[str], optional
            Read only selected items, by number (0-based), or by name
        time_steps: int or list[int], optional
            Read only selected time_steps
        elements: list[int], optional
            Read only selected element ids   

        Returns
        -------
        Dataset
            A dataset with data dimensions [t,elements]

        Examples
        --------
        >>> dfsu.read()
        <mikeio.DataSet>
        Dimensions: (9, 884)
        Time: 1985-08-06 07:00:00 - 1985-08-07 03:00:00
        Items:
        0:  Surface elevation <Surface Elevation> (meter)
        1:  U velocity <u velocity component> (meter per sec)
        2:  V velocity <v velocity component> (meter per sec)
        3:  Current speed <Current Speed> (meter per sec)
        >>> dfsu.read(time_steps="1985-08-06 12:00,1985-08-07 00:00")
        <mikeio.DataSet>
        Dimensions: (5, 884)
        Time: 1985-08-06 12:00:00 - 1985-08-06 22:00:00
        Items:
        0:  Surface elevation <Surface Elevation> (meter)
        1:  U velocity <u velocity component> (meter per sec)
        2:  V velocity <v velocity component> (meter per sec)
        3:  Current speed <Current Speed> (meter per sec)
        """

        # Open the dfs file for reading
        # self._read_dfsu_header(self._filename)
        dfs = DfsuFile.Open(self._filename)
        # time may have changes since we read the header
        # (if engine is continuously writing to this file)
        self._n_timesteps = dfs.NumberOfTimeSteps
        # TODO: add more checks that this is actually still the same file
        # (could have been replaced in the meantime)

        # NOTE. Item numbers are base 0 (everything else in the dfs is base 0)
        # n_items = self.n_items #safe_length(dfs.ItemInfo)

        nt = self.n_timesteps  # .NumberOfTimeSteps

        items, item_numbers, time_steps = get_valid_items_and_timesteps(
            self, items, time_steps
        )

        n_items = len(item_numbers)

        if elements is None:
            n_elems = self.n_elements
            n_nodes = self.n_nodes
        else:
            node_ids, _ = self._get_nodes_and_table_for_elements(elements)
            n_elems = len(elements)
            n_nodes = len(node_ids)

        deletevalue = self.deletevalue

        data_list = []

        item0_is_node_based = False
        for item in range(n_items):
            # Initialize an empty data block
            if item == 0 and items[item].name == "Z coordinate":
                item0_is_node_based = True
                data = np.ndarray(shape=(len(time_steps), n_nodes), dtype=self._dtype)
            else:
                data = np.ndarray(shape=(len(time_steps), n_elems), dtype=self._dtype)
            data_list.append(data)

        t_seconds = np.zeros(len(time_steps), dtype=float)

        for i in range(len(time_steps)):
            it = time_steps[i]
            for item in range(n_items):

                itemdata = dfs.ReadItemTimeStep(item_numbers[item] + 1, it)

                src = itemdata.Data

                d = to_numpy(src)

                d[d == deletevalue] = np.nan

                if elements is not None:
                    if item == 0 and item0_is_node_based:
                        d = d[node_ids]
                    else:
                        d = d[elements]

                data_list[item][i, :] = d

            t_seconds[i] = itemdata.Time

        time = [self.start_time + timedelta(seconds=tsec) for tsec in t_seconds]

        dfs.Close()
        return Dataset(data_list, time, items)

    def write_header(
        self, filename, start_time=None, dt=None, items=None, elements=None, title=None,
    ):
        """Write the header of a new dfsu file

            Parameters
            -----------
            filename: str
                full path to the new dfsu file
            start_time: datetime, optional
                start datetime, default is datetime.now()
            dt: float, optional
                The time step (in seconds)
            items: list[ItemInfo], optional
            elements: list[int], optional
                write only these element ids to file
            title: str
                title of the dfsu file. Default is blank.

            Examples
            --------
            >>> msh = Mesh("foo.mesh")
            >>> n_elements = msh.n_elements
            >>> dfs = Dfsu(meshfilename)
            >>> nt = 1000
            >>> n_items = 10
            >>> items = [ItemInfo(f"Item {i+1}") for i in range(n_items)]
            >>> with dfs.write_header(outfilename, items=items) as f:
            >>>     for i in range(1, nt):
            >>>         data = []
            >>>         for i in range(n_items):
            >>>             d = np.random.random((1, n_elements))
            >>>             data.append(d)
            >>>             f.append(data)
            """

        return self.write(
            filename=filename,
            data=[],
            start_time=start_time,
            dt=dt,
            items=items,
            elements=elements,
            title=title,
            keep_open=True,
        )

    def write(
        self,
        filename,
        data,
        start_time=None,
        dt=None,
        items=None,
        elements=None,
        title=None,
        keep_open=False,
    ):
        """Write a new dfsu file

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
        elements: list[int], optional
            write only these element ids to file
        title: str
            title of the dfsu file. Default is blank.
        keep_open: bool, optional
            Keep file open for appending
        """

        if isinstance(data, Dataset):
            items = data.items
            start_time = data.time[0]
            if dt is None and len(data.time) > 1:
                if not data.is_equidistant:
                    raise Exception(
                        "Data is not equidistant in time. Dfsu requires equidistant temporal axis!"
                    )
                dt = (data.time[1] - data.time[0]).total_seconds()
            data = data.data

        n_items = len(data)
        n_time_steps = 0
        if n_items > 0:
            n_time_steps = np.shape(data[0])[0]

        if dt is None:
            if self.timestep is None:
                dt = 1
            else:
                dt = self.timestep  # 1 # Arbitrary if there is only a single timestep

        if start_time is None:
            if self.start_time is None:
                start_time = datetime.now()
                warnings.warn(
                    f"No start time supplied. Using current time: {start_time} as start time."
                )
            else:
                start_time = self.start_time
                warnings.warn(
                    f"No start time supplied. Using start time from source: {start_time} as start time."
                )

        if items is None:
            if n_items == 0:
                raise ValueError(
                    "Number of items unknown. Add (..., items=[ItemInfo(...)]"
                )
            items = [ItemInfo(f"Item {i+1}") for i in range(n_items)]

        if title is None:
            title = ""

        file_start_time = to_dotnet_datetime(start_time)

        # spatial subset
        if elements is None:
            geometry = self
        else:
            geometry = self.elements_to_geometry(elements)
            if (not self.is_2d) and (geometry._type == UnstructuredType.Dfsu2D):
                # redo extraction as 2d:
                print("will redo extraction in 2d!")
                geometry = self.elements_to_geometry(elements, node_layers="bottom")
                if items[0].name == "Z coordinate":
                    # get rid of z-item
                    items = items[1 : (n_items + 1)]
                    n_items = n_items - 1
                    new_data = []
                    for j in range(n_items):
                        new_data.append(data[j + 1])
                    data = new_data

        # Default filetype;
        if geometry._type == UnstructuredType.Mesh:
            # create dfs2d from mesh
            dfsu_filetype = DfsuFileType.Dfsu2D
        else:
            # TODO: if subset is slice...
            dfsu_filetype = geometry._type.value

        if dfsu_filetype != DfsuFileType.Dfsu2D:
            if items[0].name != "Z coordinate":
                raise Exception("First item must be z coordinates of the nodes!")

        xn = geometry.node_coordinates[:, 0]
        yn = geometry.node_coordinates[:, 1]

        # zn have to be Single precision??
        zn = to_dotnet_float_array(geometry.node_coordinates[:, 2])

        elem_table = []
        for j in range(geometry.n_elements):
            elem_nodes = geometry.element_table[j]
            elem_nodes = [nd + 1 for nd in elem_nodes]
            elem_table.append(elem_nodes)
        elem_table = asnetarray_v2(elem_table)

        builder = DfsuBuilder.Create(dfsu_filetype)

        builder.SetNodes(xn, yn, zn, geometry.codes)
        builder.SetElements(elem_table)
        # builder.SetNodeIds(geometry.node_ids+1)
        # builder.SetElementIds(geometry.elements+1)

        factory = DfsFactory()
        proj = factory.CreateProjection(geometry.projection_string)
        builder.SetProjection(proj)
        builder.SetTimeInfo(file_start_time, dt)
        builder.SetZUnit(eumUnit.eumUmeter)

        if dfsu_filetype != DfsuFileType.Dfsu2D:
            builder.SetNumberOfSigmaLayers(geometry.n_sigma_layers)

        for item in items:
            if item.name != "Z coordinate":
                builder.AddDynamicItem(
                    item.name, eumQuantity.Create(item.type, item.unit)
                )

        try:
            self._dfs = builder.CreateFile(filename)
        except IOError:
            print("cannot create dfsu file: ", filename)

        deletevalue = self._dfs.DeleteValueFloat

        try:
            # Add data for all item-timesteps, copying from source
            for i in range(n_time_steps):
                for item in range(n_items):
                    d = data[item][i, :]
                    d[np.isnan(d)] = deletevalue
                    darray = to_dotnet_float_array(d)
                    self._dfs.WriteItemTimeStepNext(0, darray)
            if not keep_open:
                self._dfs.Close()
            else:
                return self

        except Exception as e:
            print(e)
            self._dfs.Close()
            os.remove(filename)

    def append(self, data):
        """Append to a dfsu file opened with `write(...,keep_open=True)`

        Parameters
        -----------
        data: list[np.array]
        """

        deletevalue = self._dfs.DeleteValueFloat
        n_items = len(data)
        n_time_steps = np.shape(data[0])[0]
        for i in range(n_time_steps):
            for item in range(n_items):
                d = data[item][i, :]
                d[np.isnan(d)] = deletevalue
                darray = to_dotnet_float_array(d)
                self._dfs.WriteItemTimeStepNext(0, darray)

    def close(self):
        "Finalize write for a dfsu file opened with `write(...,keep_open=True)`"
        self._dfs.Close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self._dfs.Close()

    def to_mesh(self, outfilename):
        """write object to mesh file

        Parameters
        ----------
        outfilename : str
            path to file to be written
        """
        if self.is_2d:
            _ = self.element_table  # make sure element table has been constructured
            geometry = self
        else:
            geometry = self.to_2d_geometry()
            # TODO: print warning if sigma-z
        Mesh._geometry_to_mesh(outfilename, geometry)


class Mesh(_UnstructuredFile):
    def __init__(self, filename):
        super().__init__()
        self._filename = filename
        self._read_header(filename)
        self._n_timesteps = None
        self._n_items = None
        self._n_layers = None
        self._n_sigma = None
        self._type = UnstructuredType.Mesh

    def set_z(self, z):
        """Change the depth by setting the z value of each node

        Parameters
        ----------
        z : np.array(float)
            new z value at each node
        """
        if len(z) != self.n_nodes:
            raise Exception(f"z must have length of nodes ({self.n_nodes})")
        self._nc[:, 2] = z
        self._ec = None

    def set_codes(self, codes):
        """Change the code values of the nodes

        Parameters
        ----------
        codes : list(int)
            code of each node
        """
        if len(codes) != self.n_nodes:
            raise Exception(f"codes must have length of nodes ({self.n_nodes})")
        self._codes = codes
        self._valid_codes = None

    def write(self, outfilename, elements=None):
        """write mesh to file (will overwrite if file exists)

        Parameters
        ----------
        outfilename : str
            path to file
        elements : list(int)
            list of element ids (subset) to be saved to new mesh
        """
        builder = MeshBuilder()

        if elements is None:
            geometry = self
            quantity = self._source.EumQuantity
            elem_table = self._source.ElementTable
        else:
            geometry = self.elements_to_geometry(elements)
            quantity = eumQuantity.Create(EUMType.Bathymetry, EUMUnit.meter)
            elem_table = geometry._element_table_to_dotnet()

        nc = geometry.node_coordinates
        builder.SetNodes(nc[:, 0], nc[:, 1], nc[:, 2], geometry.codes)

        builder.SetElements(elem_table)
        builder.SetProjection(geometry.projection_string)
        builder.SetEumQuantity(quantity)

        newMesh = builder.CreateMesh()
        newMesh.Write(outfilename)

    def plot_boundary_nodes(self, boundary_names=None):
        """
        Plot mesh boundary nodes and their codes
        """
        import matplotlib.pyplot as plt

        nc = self.node_coordinates
        c = self.codes

        if boundary_names is not None:
            if len(self.boundary_codes) != len(boundary_names):
                raise Exception(
                    f"Number of boundary names ({len(boundary_names)}) inconsistent with number of boundaries ({len(self.boundary_codes)})"
                )
            user_defined_labels = dict(zip(self.boundary_codes, boundary_names))

        fig, ax = plt.subplots()
        for code in self.boundary_codes:
            xn = nc[c == code, 0]
            yn = nc[c == code, 1]
            if boundary_names is None:
                label = f"Code {code}"
            else:
                label = user_defined_labels[code]
            plt.plot(xn, yn, ".", label=label)

        plt.legend()
        plt.title("Boundary nodes")
        ax.set_xlim(nc[:, 0].min(), nc[:, 0].max())
        ax.set_ylim(nc[:, 1].min(), nc[:, 1].max())

    @staticmethod
    def _geometry_to_mesh(outfilename, geometry):

        builder = MeshBuilder()

        nc = geometry.node_coordinates
        builder.SetNodes(nc[:, 0], nc[:, 1], nc[:, 2], geometry.codes)
        # builder.SetNodeIds(geometry.node_ids+1)
        # builder.SetElementIds(geometry.elements+1)
        builder.SetElements(geometry._element_table_to_dotnet())
        builder.SetProjection(geometry.projection_string)
        quantity = eumQuantity.Create(EUMType.Bathymetry, EUMUnit.meter)
        builder.SetEumQuantity(quantity)
        newMesh = builder.CreateMesh()
        newMesh.Write(outfilename)
