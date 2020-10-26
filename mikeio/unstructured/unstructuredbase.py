import numpy as np
import warnings

from scipy.spatial import cKDTree

from DHI.Generic.MikeZero.DFS.dfsu import DfsuUtil

from ..dotnet import (
    asNumpyArray,
    asnetarray_v2,
)
from ..spatial import Grid2D
from ..interpolation import get_idw_interpolant
from .plot import UnstructuredPlotter

from enum import IntEnum


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
            warnings.warn(
                "Could not determine if domain contains points. Failed to convert to_shapely()"
            )
            return None

        cnts = mp.Path(domain.exterior).contains_points(points)
        for interj in domain.interiors:
            in_hole = mp.Path(interj).contains_points(points)
            cnts = np.logical_and(cnts, ~in_hole)
        return cnts

    def _create_tree2d(self):
        xy = self.geometry2d.element_coordinates[:, :2]
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
        coords = self.geometry2d.node_coordinates  # node_ or element_
        small = 1e-10
        x0 = coords[:, 0].min() + small
        y0 = coords[:, 1].min() + small
        x1 = coords[:, 0].max() - small
        y1 = coords[:, 1].max() - small
        bbox = [x0, y0, x1, y1]
        return Grid2D(bbox=bbox, dxdy=dxdy, shape=shape)

    def get_2d_interpolant(self, xy, n_nearest: int = 1, extrapolate=False):
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
        weights = None

        if n_nearest == 1:
            weights = np.ones(distances.shape)
            if not extrapolate:
                weights[~self.contains(xy)] = np.nan
        elif n_nearest > 1:
            weights = get_idw_interpolant(distances)
            if not extrapolate:
                weights[~self.contains(xy), :] = np.nan
        else:
            ValueError("n_nearest must be at least 1")

        return elem_ids, weights

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
                earth_radius = 6366707.0 * np.pi / 180.0

                # Y on element centers
                Ye = np.sum(ycoords[:n_nodes]) / n_nodes
                cosYe = np.cos(np.deg2rad(Ye))

                abx = earth_radius * abx * cosYe
                aby = earth_radius * aby
                acx = earth_radius * acx * cosYe
                acy = earth_radius * acy
                if isquad:
                    adx = earth_radius * adx * cosYe
                    ady = earth_radius * ady

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
            # print("Object has no layers: cannot return geometry2d")
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

        plotter = UnstructuredPlotter(self)
        plotter.plot(
            z,
            elements,
            plot_type,
            title,
            label,
            cmap,
            vmin,
            vmax,
            levels,
            n_refinements,
            show_mesh,
            show_outline,
            figsize,
            ax,
        )


# TODO Not so sure about this
def get_nodes_from_source(source):
    xn = asNumpyArray(source.X)
    yn = asNumpyArray(source.Y)
    zn = asNumpyArray(source.Z)

    nc = np.column_stack([xn, yn, zn])
    codes = np.array(list(source.Code))
    n_nodes = source.NumberOfNodes
    node_ids = np.array(list(source.NodeIds)) - 1

    return nc, codes, n_nodes, node_ids


def get_elements_from_source(source):
    n_elements = source.NumberOfElements
    element_table_dotnet = source.ElementTable
    element_ids = np.array(list(source.ElementIds)) - 1

    return n_elements, element_table_dotnet, element_ids
