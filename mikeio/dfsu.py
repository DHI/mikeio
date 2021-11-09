from logging import warn
import os
from collections import namedtuple
import pandas as pd
import pathlib
import warnings
import numpy as np
from datetime import datetime, timedelta
from scipy.spatial import cKDTree
import tempfile
from tqdm import trange
import typing

from mikecore.eum import eumUnit, eumQuantity
from mikecore.DfsFactory import DfsFactory
from mikecore.DfsuBuilder import DfsuBuilder
from mikecore.DfsuFile import DfsuFile
from mikecore.DfsuFile import DfsuFileType, DfsuUtil

from mikecore.MeshFile import MeshFile
from mikecore.MeshBuilder import MeshBuilder

from .dfsutil import _get_item_info, _valid_item_numbers, _valid_timesteps
from .dataset import Dataset
from .dfs0 import Dfs0
from .dfs2 import Dfs2
from .eum import ItemInfo, EUMType, EUMUnit

from .spatial import Grid2D
from .interpolation import get_idw_interpolant, interp2d
from .custom_exceptions import InvalidGeometry
from .base import EquidistantTimeSeries


class _UnstructuredGeometry:
    # THIS CLASS KNOWS NOTHING ABOUT MIKE FILES!

    def __init__(self) -> None:
        self._type = None  # -1: mesh, 0: 2d-dfsu, 4:dfsu3dsigma, ...
        self._projstr = None

        self._n_nodes = None
        self._n_elements = None
        self._nc = None
        self._ec = None
        self._codes = None
        self._valid_codes = None
        self._element_ids = None
        self._node_ids = None
        self._element_table = None
        self._element_table_mikecore = None

        self._top_elems = None
        self._n_layers_column = None
        self._bot_elems = None
        self._n_layers = None
        self._n_sigma = None

        self._geom2d = None
        self._e2_e3_table = None
        self._2d_ids = None
        self._layer_ids = None

        self._shapely_domain_obj = None
        self._tree2d = None

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
        """Type name, e.g. Mesh, Dfsu2D"""
        return self._type.name if self._type else "Mesh"

    @property
    def n_nodes(self):
        """Number of nodes"""
        return self._n_nodes

    @property
    def node_coordinates(self):
        """Coordinates (x,y,z) of all nodes"""
        return self._nc

    @property
    def node_ids(self):
        return self._node_ids

    @property
    def n_elements(self):
        """Number of elements"""
        return self._n_elements

    @property
    def element_ids(self):
        return self._element_ids

    @property
    def codes(self):
        """Node codes of all nodes (0=water, 1=land, 2...=open boundaries)"""
        return self._codes

    @codes.setter
    def codes(self, v):
        if len(v) != self.n_nodes:
            raise ValueError(f"codes must have length of nodes ({self.n_nodes})")
        self._codes = np.array(v, dtype=np.int32)
        self._valid_codes = None

    @property
    def valid_codes(self):
        """Unique list of node codes"""
        if self._valid_codes is None:
            self._valid_codes = list(set(self.codes))
        return self._valid_codes

    @property
    def boundary_codes(self):
        """Unique list of boundary codes"""
        return [code for code in self.valid_codes if code > 0]

    @property
    def projection_string(self):
        """The projection string"""
        return self._projstr

    @property
    def is_geo(self):
        """Are coordinates geographical (LONG/LAT)?"""
        return self._projstr == "LONG/LAT"

    @property
    def is_local_coordinates(self):
        """Are coordinates relative (NON-UTM)?"""
        return self._projstr == "NON-UTM"

    @property
    def element_table(self):
        """Element to node connectivity"""
        if (self._element_table is None) and (self._element_table_mikecore is not None):
            self._element_table = self._get_element_table_from_mikecore()
        return self._element_table

    @property
    def max_nodes_per_element(self):
        """The maximum number of nodes for an element"""
        maxnodes = 0
        for local_nodes in self.element_table:
            n = len(local_nodes)
            if n > maxnodes:
                maxnodes = n
        return maxnodes

    @property
    def is_2d(self):
        """Type is either mesh or Dfsu2D (2 horizontal dimensions)"""
        return self._type in (DfsuFileType.Dfsu2D, None)  # Todo mesh

    @property
    def is_tri_only(self):
        """Does the mesh consist of triangles only?"""
        return self.max_nodes_per_element == 3 or self.max_nodes_per_element == 6

    _boundary_polylines = None

    @property
    def boundary_polylines(self):
        """Lists of closed polylines defining domain outline"""
        if self._boundary_polylines is None:
            self._boundary_polylines = self._get_boundary_polylines()
        return self._boundary_polylines

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

    def _get_element_table_from_mikecore(self):
        # Note: this can tak 10-20 seconds for large dfsu3d!

        elem_tbl = self._element_table_mikecore.copy()
        for j in range(self.n_elements):
            elem_tbl[j] = self._element_table_mikecore[j] - 1
        return elem_tbl

    def _element_table_to_mikecore(self, elem_table=None):
        if elem_table is None:
            elem_table = self._element_table
        new_elem_table = []
        n_elements = len(elem_table)
        for j in range(n_elements):
            elem_nodes = elem_table[j]
            elem_nodes = [nd + 1 for nd in elem_nodes]  # make 1-based
            new_elem_table.append(np.array(elem_nodes))
        return new_elem_table

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
                geometry_type = DfsuFileType.Dfsu2D
            else:
                geometry_type = DfsuFileType.Dfsu3DSigma
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
        """export elements to new flexible file geometry

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
                self._type == DfsuFileType.Dfsu3DSigma
                or self._type == DfsuFileType.Dfsu3DSigmaZ
            ) and n_layers == 1:
                # If source is 3d, but output only has 1 layer
                # then change type to 2d
                geom._type = DfsuFileType.Dfsu2D
                geom._n_layers = None
                if node_layers == "all":
                    warnings.warn(
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
                    self._type == DfsuFileType.DfsuVerticalProfileSigmaZ
                    or self._type == DfsuFileType.Dfsu3DSigmaZ
                ) and n_layers == geom._n_sigma:
                    # TODO fix this
                    geom._type = DfsuFileType.Dfsu3DSigma

                geom._top_elems = geom._get_top_elements_from_coordinates()

        return geom

    def _get_top_elements_from_coordinates(self, ec=None):
        """Get list of top element ids based on element coordinates"""
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
        if self.is_2d:
            return self

        # extract information for selected elements
        elem_ids = self.bottom_elements
        if self._type == DfsuFileType.Dfsu3DSigmaZ:
            # for z-layers nodes will not match on neighboring elements!
            elem_ids = self.top_elements

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

        # TODO how to handle Mesh filetype
        geom._type = None  # DfsuFileType.Mesh

        geom._reindex()

        # Fix z-coordinate for sigma-z:
        if self._type == DfsuFileType.Dfsu3DSigmaZ:
            zn = geom.node_coordinates[:, 2].copy()
            for j, elem_nodes in enumerate(geom.element_table):
                elem_nodes3d = self.element_table[self.bottom_elements[j]]
                for jn in range(len(elem_nodes)):
                    znj_3d = self.node_coordinates[elem_nodes3d[jn], 2]
                    zn[elem_nodes[jn]] = min(zn[elem_nodes[jn]], znj_3d)
            geom.node_coordinates[:, 2] = zn

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
        """Center coordinates of each element"""
        if self._ec is None:
            self._ec = self.calc_element_coordinates()
        return self._ec

    def calc_element_coordinates(self, elements=None, zn=None):
        """Calculates the coordinates of the center of each element.

        Only necessary for dynamic vertical coordinates,
        otherwise use the property *element_coordinates* instead

        Parameters
        ----------
        elements : np.array(int), optional
            element ids of selected elements
        zn : np.array(float), optional
            only the z-coodinates of the nodes

        Examples
        --------
        >>> elem_ids = dfs.find_nearest_profile_elements(x0, y0)
        >>> ds = dfs.read(items=['Z coordinate','Temperature'], elements=elem_ids)
        >>> ec_dyn = dfs.calc_element_coordinates(elements=elem_ids, zn=ds['Z coordinate'][0,:])
        >>> plt.plot(ds['Temperature'][0, :], ec_dyn[:,2])

        Returns
        -------
        np.array
            x,y,z of each element
        """
        node_coordinates = self._nc

        element_table = self.element_table
        if elements is not None:
            element_table = element_table[elements]
        if zn is not None:
            node_coordinates = node_coordinates.copy()
            if len(zn) == len(node_coordinates[:, 2]):
                node_coordinates[:, 2] = zn
            else:
                # assume that user wants to find coords on a subset of points
                idx = np.unique(np.hstack(element_table))
                node_coordinates[idx, 2] = zn

        n_elements = len(element_table)
        ec = np.empty([n_elements, 3])

        # pre-allocate for speed
        maxnodes = 4 if self.is_2d else 8
        idx = np.zeros(maxnodes, dtype=int)
        xcoords = np.zeros([maxnodes, n_elements])
        ycoords = np.zeros([maxnodes, n_elements])
        zcoords = np.zeros([maxnodes, n_elements])
        nnodes_per_elem = np.zeros(n_elements)

        for j in range(n_elements):
            nodes = element_table[j]
            nnodes = len(nodes)
            nnodes_per_elem[j] = nnodes
            for i in range(nnodes):
                idx[i] = nodes[i]  # - 1

            xcoords[:nnodes, j] = node_coordinates[idx[:nnodes], 0]
            ycoords[:nnodes, j] = node_coordinates[idx[:nnodes], 1]
            zcoords[:nnodes, j] = node_coordinates[idx[:nnodes], 2]

        ec[:, 0] = np.sum(xcoords, axis=0) / nnodes_per_elem
        ec[:, 1] = np.sum(ycoords, axis=0) / nnodes_per_elem
        ec[:, 2] = np.sum(zcoords, axis=0) / nnodes_per_elem

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

        points = np.atleast_2d(points)

        exterior = self.boundary_polylines.exteriors[0]
        cnts = mp.Path(exterior.xy).contains_points(points)

        if self.boundary_polylines.n_exteriors > 1:
            # in case of several dis-joint outer domains
            for exterior in self.boundary_polylines.exteriors[1:]:
                in_domain = mp.Path(exterior.xy).contains_points(points)
                cnts = np.logical_or(cnts, in_domain)

        # subtract any holes
        for interior in self.boundary_polylines.interiors:
            in_hole = mp.Path(interior.xy).contains_points(points)
            cnts = np.logical_and(cnts, ~in_hole)

        return cnts

    def get_overset_grid(self, dx=None, dy=None, shape=None, buffer=None):
        """get a 2d grid that covers the domain by specifying spacing or shape

        Parameters
        ----------
        dx : float or (float, float), optional
            grid resolution in x-direction (or in x- and y-direction)
        dy : float, optional
            grid resolution in y-direction
        shape : (int, int), optional
            tuple with nx and ny describing number of points in each direction
            one of them can be None, in which case the value will be inferred
        buffer : float, optional
            positive to make the area larger, default=0
            can be set to a small negative value to avoid NaN
            values all around the domain.

        Returns
        -------
        <mikeio.Grid2D>
            2d grid
        """
        nc = self.geometry2d.node_coordinates
        bbox = Grid2D.xy_to_bbox(nc, buffer=buffer)
        return Grid2D(bbox=bbox, dx=dx, dy=dy, shape=shape)

    def get_2d_interpolant(
        self, xy, n_nearest: int = 1, extrapolate=False, p=2, radius=None
    ):
        """IDW interpolant for list of coordinates

        Parameters
        ----------
        xy : array-like
            x,y coordinates of new points
        n_nearest : int, optional
            [description], by default 1
        extrapolate : bool, optional
            allow , by default False
        p : float, optional
            power of inverse distance weighting, default=2
        radius: float, optional
            an alternative to extrapolate=False,
            only include elements within radius

        Returns
        -------
        (np.array, np.array)
            element ids and weights
        """
        xy = np.atleast_2d(xy)
        ids, dists = self._find_n_nearest_2d_elements(xy, n=n_nearest)
        weights = None

        if n_nearest == 1:
            weights = np.ones(dists.shape)
            if not extrapolate:
                weights[~self.contains(xy)] = np.nan
        elif n_nearest > 1:
            weights = get_idw_interpolant(dists, p=p)
            if not extrapolate:
                weights[~self.contains(xy), :] = np.nan
        else:
            ValueError("n_nearest must be at least 1")

        if radius is not None:
            idx = np.where(dists > radius)[0]
            weights[idx] = np.nan

        return ids, weights

    def interp2d(self, data, elem_ids, weights=None, shape=None):
        """interp spatially in data (2d only)

        Parameters
        ----------
        data : ndarray or list(ndarray)
            dfsu data
        elem_ids : ndarray(int)
            n sized array of 1 or more element ids used for interpolation
        weights : ndarray(float), optional
            weights with same size as elem_ids used for interpolation
        shape: tuple, optional
            reshape output

        Returns
        -------
        ndarray or list(ndarray)
            spatially interped data

        Examples
        --------
        >>> ds = dfsu.read()
        >>> g = dfs.get_overset_grid(shape=(50,40))
        >>> elem_ids, weights = dfs.get_2d_interpolant(g.xy)
        >>> dsi = dfs.interp2d(ds, elem_ids, weights)
        """
        return interp2d(data, elem_ids, weights, shape)

    def _create_tree2d(self):
        xy = self.geometry2d.element_coordinates[:, :2]
        self._tree2d = cKDTree(xy)

    def _find_n_nearest_2d_elements(self, x, y=None, n=1):
        if n > self.geometry2d.n_elements:
            raise ValueError(
                f"Cannot find {n} nearest! Number of 2D elements: {self.geometry2d.n_elements}"
            )

        if self._tree2d is None:
            self._create_tree2d()

        if y is None:
            p = x
            if (not np.isscalar(x)) and (np.ndim(x) == 2):
                p = x[:, 0:2]
        else:
            p = np.array((x, y)).T
        d, elem_id = self._tree2d.query(p, k=n)
        return elem_id, d

    def _find_3d_from_2d_points(self, elem2d, z=None, layer=None):

        was_scalar = np.isscalar(elem2d)
        if was_scalar:
            elem2d = np.array([elem2d])
        else:
            orig_shape = elem2d.shape
            elem2d = np.reshape(elem2d, (elem2d.size,))

        if (layer is None) and (z is None):
            # return top element
            idx = self.top_elements[elem2d]

        elif layer is None:
            idx = np.zeros_like(elem2d)
            if np.isscalar(z):
                z = z * np.ones_like(elem2d, dtype=float)
            elem3d = self.e2_e3_table[elem2d]
            for j, row in enumerate(elem3d):
                zc = self.element_coordinates[row, 2]
                d3d = np.abs(z[j] - zc)
                idx[j] = row[d3d.argsort()[0]]

        elif z is None:
            if 0 <= layer <= self.n_z_layers - 1:
                idx = np.zeros_like(elem2d)
                elem3d = self.e2_e3_table[elem2d]
                for j, row in enumerate(elem3d):
                    try:
                        layer_ids = self.layer_ids[row]
                        id = row[list(layer_ids).index(layer)]
                        idx[j] = id
                    except:
                        print(f"Layer {layer} not present for 2d element {elem2d[j]}")
            else:
                # sigma layer
                idx = self.get_layer_elements(layer=layer)[elem2d]

        else:
            raise ValueError("layer and z cannot both be supplied!")

        if was_scalar:
            idx = idx[0]
        else:
            idx = np.reshape(idx, orig_shape)

        return idx

    def find_nearest_element(self, x, y, z=None, layer=None, n_nearest=1):
        warnings.warn("OBSOLETE! method name changed to find_nearest_elements")
        return self.find_nearest_elements(x, y, z, layer, n_nearest)

    def find_nearest_elements(
        self, x, y=None, z=None, layer=None, n_nearest=1, return_distances=False
    ):
        """Find index of nearest elements (optionally for a list)

        Parameters
        ----------

        x: float or array(float)
            X coordinate(s) (easting or longitude)
        y: float or array(float)
            Y coordinate(s) (northing or latitude)
        z: float or array(float), optional
            Z coordinate(s)  (vertical coordinate, positive upwards)
            If not provided for a 3d file, the surface element is returned
        layer: int, optional
            Search in a specific layer only (3D files only)
            Either z or layer (0-based) can be provided for a 3D file
        n_nearest : int, optional
            return this many (horizontally) nearest points for
            each coordinate set, default=1
        return_distances : bool, optional
            should the horizontal distances to each point be returned?
            default=False

        Returns
        -------
        np.array
            element ids of nearest element(s)
        np.array, optional
            horizontal distances

        Examples
        --------
        >>> id = dfs.find_nearest_elements(3, 4)
        >>> ids = dfs.find_nearest_elements([3, 8], [4, 6])
        >>> ids = dfs.find_nearest_elements(xy)
        >>> ids = dfs.find_nearest_elements(3, 4, n_nearest=4)
        >>> ids, d = dfs.find_nearest_elements(xy, return_distances=True)

        >>> ids = dfs.find_nearest_elements(3, 4, z=-3)
        >>> ids = dfs.find_nearest_elements(3, 4, layer=4)
        >>> ids = dfs.find_nearest_elements(xyz)
        >>> ids = dfs.find_nearest_elements(xyz, n_nearest=3)
        """
        idx, d2d = self._find_n_nearest_2d_elements(x, y, n=n_nearest)

        if not self.is_2d:
            if self._use_third_col_as_z(x, z, layer):
                z = x[:, 2]
            idx = self._find_3d_from_2d_points(idx, z=z, layer=layer)

        if return_distances:
            return idx, d2d

        return idx

    def _use_third_col_as_z(self, x, z, layer):
        return (
            (z is None)
            and (layer is None)
            and (not np.isscalar(x))
            and (np.ndim(x) == 2)
            and (x.shape[1] >= 3)
        )

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
            raise InvalidGeometry("Object is 2d. Cannot get_nearest_profile")
        else:
            elem2d, _ = self._find_n_nearest_2d_elements(x, y)
            elem3d = self.e2_e3_table[elem2d]
            return elem3d

    def get_element_area(self):
        """Calculate the horizontal area of each element.

        Returns
        -------
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
        """The 2d geometry for a 3d object"""
        if self._n_layers is None:
            return self
        if self._geom2d is None:
            self._geom2d = self.to_2d_geometry()
        return self._geom2d

    @property
    def e2_e3_table(self):
        """The 2d-to-3d element connectivity table for a 3d object"""
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
        """The associated 2d element id for each 3d element"""
        if self._n_layers is None:
            raise InvalidGeometry("Object has no layers: cannot return elem2d_ids")
            # or return self._2d_ids ??

        if self._2d_ids is None:
            res = self._get_2d_to_3d_association()
            self._e2_e3_table = res[0]
            self._2d_ids = res[1]
            self._layer_ids = res[2]
        return self._2d_ids

    @property
    def layer_ids(self):
        """The layer number (0=bottom, 1, 2, ...) for each 3d element"""
        if self._n_layers is None:
            raise InvalidGeometry("Object has no layers: cannot return layer_ids")
        if self._layer_ids is None:
            res = self._get_2d_to_3d_association()
            self._e2_e3_table = res[0]
            self._2d_ids = res[1]
            self._layer_ids = res[2]
        return self._layer_ids

    @property
    def n_layers(self):
        """Maximum number of layers"""
        return self._n_layers

    @property
    def n_sigma_layers(self):
        """Number of sigma layers"""
        return self._n_sigma

    @property
    def n_z_layers(self):
        """Maximum number of z-layers"""
        if self._n_layers is None:
            return None
        return self._n_layers - self._n_sigma

    @property
    def top_elements(self):
        """List of 3d element ids of surface layer"""
        if self._n_layers is None:
            print("Object has no layers: cannot find top_elements")
            return None
        elif (self._top_elems is None) and (self._source is not None):
            # note: if subset of elements is selected then this cannot be done!
            self._top_elems = np.array(
                DfsuUtil.FindTopLayerElements(self._source.ElementTable)
            )
        return self._top_elems

    @property
    def n_layers_per_column(self):
        """List of number of layers for each column"""
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
        """List of 3d element ids of bottom layer"""
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
            layer between 0 (bottom) and n_layers-1 (top)
            (can also be negative counting from -1 at the top layer)

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
            raise InvalidGeometry("Object has no layers: cannot get_layer_elements")

        if layer < (-n_lay) or layer >= n_lay:
            raise Exception(
                f"Layer {layer} not allowed; must be between -{n_lay} and {n_lay-1}"
            )

        if layer < 0:
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
        # layer_ids = 0, 1, 2...
        global_layer_ids = np.arange(self.n_layers)
        for j in range(n2d):
            col = list(range(botid[j], topid[j] + 1))

            e2_to_e3.append(col)
            for jj in col:
                index2d.append(j)

            n_local_layers = len(col)
            local_layers = global_layer_ids[-n_local_layers:]
            for ll in local_layers:
                layerid.append(ll)

        e2_to_e3 = np.array(e2_to_e3, dtype=object)
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
            for i in np.unique(
                elem_table.reshape(
                    -1,
                )
            )
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

    def _Get_2DVertical_elements(self):
        if (self._type == DfsuFileType.DfsuVerticalProfileSigmaZ) or (
            self._type == DfsuFileType.DfsuVerticalProfileSigma
        ):
            elements = [
                list(self._source.ElementTable[i])
                for i in range(len(list(self._source.ElementTable)))
            ]
            return np.asarray(elements) - 1

    def plot_vertical_profile(
        self, values, time_step=None, cmin=None, cmax=None, label="", **kwargs
    ):
        """
        Plot unstructured vertical profile

        Parameters
        ----------
        values: np.array
            value for each element to plot
        timestep: int, optional
            the timestep that fits with the data to get correct vertical
            positions, default: use static vertical positions
        cmin: real, optional
            lower bound of values to be shown on plot, default:None
        cmax: real, optional
            upper bound of values to be shown on plot, default:None
        title: str, optional
            axes title
        label: str, optional
            colorbar label
        cmap: matplotlib.cm.cmap, optional
            colormap, default viridis
        figsize: (float, float), optional
            specify size of figure
        ax: matplotlib.axes, optional
            Adding to existing axis, instead of creating new fig

        Returns
        -------
        <matplotlib.axes>
        """
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection

        nc = self.node_coordinates
        x_coordinate = np.hypot(nc[:, 0], nc[:, 1])
        if time_step is None:
            y_coordinate = nc[:, 2]
        else:
            y_coordinate = self.read()[0][time_step, :]

        elements = self._Get_2DVertical_elements()

        # plot in existing or new axes?
        if "ax" in kwargs:
            ax = kwargs["ax"]
        else:
            figsize = None
            if "figsize" in kwargs:
                figsize = kwargs["figsize"]
            _, ax = plt.subplots(figsize=figsize)

        yz = np.c_[x_coordinate, y_coordinate]
        verts = yz[elements]

        if "cmap" in kwargs:
            cmap = kwargs["cmap"]
        else:
            cmap = "jet"
        pc = PolyCollection(verts, cmap=cmap)

        if cmin is None:
            cmin = np.nanmin(values)
        if cmax is None:
            cmax = np.nanmax(values)
        pc.set_clim(cmin, cmax)

        plt.colorbar(pc, ax=ax, label=label, orientation="vertical")
        pc.set_array(values)

        if "edge_color" in kwargs:
            edge_color = kwargs["edge_color"]
        else:
            edge_color = None
        pc.set_edgecolor(edge_color)

        ax.add_collection(pc)
        ax.autoscale()

        if "title" in kwargs:
            ax.set_title(kwargs["title"])

        return ax

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
        levels=None,
        n_refinements=0,
        show_mesh=True,
        show_outline=True,
        figsize=None,
        ax=None,
        add_colorbar=True,
    ):
        """
        Plot unstructured data and/or mesh, mesh outline

        Parameters
        ----------
        z: np.array or a Dataset with a single item, optional
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
        add_olorbar: bool
            Add colorbar to plot, default True

        Returns
        -------
        <matplotlib.axes>

        Examples
        --------
        >>> dfs = Dfsu("HD2D.dfsu")
        >>> dfs.plot() # bathymetry
        >>> ds = dfs.read(items="Surface elevation", time_steps=0)
        >>> ds.shape
        (1, 884)
        >>> ds.n_items
        1
        >>> dfs.plot(z=ds) # plot surface elevation
        """

        import matplotlib.cm as cm
        import matplotlib.colors as mplc
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
            if len(z) == 1:  # if single-item Dataset
                z = z[0].copy()
            if len(z) != ne:
                z = np.squeeze(z).copy()  # handles single time step
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
        cmap_norm = None
        cmap_ScMappable = None
        if ("only" not in plot_type) and (levels is not None):
            if np.isscalar(levels):
                n_levels = levels
                levels = np.linspace(vmin, vmax, n_levels)
            else:
                n_levels = len(levels)
                vmin = min(levels)
                vmax = max(levels)

            levels = np.array(levels)
            levels_equidistant = all(np.diff(levels) == np.diff(levels)[0])
            # if not levels_equidistant:  # (isinstance(cmap, mplc.ListedColormap)) or (
            if (not isinstance(cmap, str)) and (not levels_equidistant):
                print("ScalarMappable")
                # cmap = mplc.Colormap(cmap)
                cmap_norm = mplc.BoundaryNorm(levels, cmap.N)
                cmap_ScMappable = cm.ScalarMappable(cmap=cmap, norm=cmap_norm)
        if ("contour" in plot_type) and (levels is None):
            levels = 10
            n_levels = 10

        cbar_extend = self._cbar_extend(z, vmin, vmax)

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
                    patches,
                    cmap=cmap,
                    norm=cmap_norm,
                    edgecolor=mesh_col,
                    linewidths=0.4,
                )
            else:
                fig_obj = PatchCollection(
                    patches,
                    cmap=cmap,
                    norm=cmap_norm,
                    edgecolor="face",
                    alpha=None,
                    linewidths=None,
                )

            fig_obj.set_array(z)
            fig_obj.set_clim(vmin, vmax)
            ax.add_collection(fig_obj)

            if add_colorbar:
                cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
                cmap_sm = cmap_ScMappable if cmap_ScMappable else fig_obj
                plt.colorbar(cmap_sm, label=label, cax=cax, extend=cbar_extend)

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
                if cmap_norm is None:
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
                else:
                    fig_obj = ax.tripcolor(
                        triang,
                        zn,
                        edgecolors="face",
                        cmap=cmap,
                        norm=cmap_norm,
                        linewidths=0.3,
                        shading="gouraud",
                    )

                if add_colorbar:
                    cax = make_axes_locatable(ax).append_axes(
                        "right", size="5%", pad=0.05
                    )
                    cmap_sm = cmap_ScMappable if cmap_ScMappable else fig_obj

                    plt.colorbar(
                        cmap_sm,
                        label=label,
                        cax=cax,
                        boundaries=levels,
                        extend=cbar_extend,
                    )

            elif plot_type == "contour" or plot_type == "contour_lines":
                ax.triplot(triang, lw=mesh_linewidth, color=mesh_col_dark)
                fig_obj = ax.tricontour(
                    triang,
                    zn,
                    levels=levels,
                    linewidths=[1.2],
                    cmap=cmap,
                    norm=cmap_norm,
                )
                ax.clabel(fig_obj, fmt="%1.2f", inline=1, fontsize=9)
                if len(label) > 0:
                    ax.set_title(label)

            elif plot_type == "contourf" or plot_type == "contour_filled":
                ax.triplot(triang, lw=mesh_linewidth, color=mesh_col)
                vbuf = 0.01 * (vmax - vmin) / n_levels
                # avoid white outside limits
                zn = np.clip(zn, vmin + vbuf, vmax - vbuf)
                fig_obj = ax.tricontourf(
                    triang,
                    zn,
                    levels=levels,
                    cmap=cmap,
                    norm=cmap_norm,
                    extend=cbar_extend,
                )

                # colorbar
                if add_colorbar:
                    cax = make_axes_locatable(ax).append_axes(
                        "right", size="5%", pad=0.05
                    )
                    if cmap_ScMappable is None:
                        plt.colorbar(fig_obj, label=label, cax=cax)
                    else:
                        plt.colorbar(
                            cmap_ScMappable,
                            label=label,
                            cax=cax,
                            ticks=levels,
                        )

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
            linwid = 1.2
            out_col = "0.4"
            for exterior in self.boundary_polylines.exteriors:
                ax.plot(*exterior.xy.T, color=out_col, linewidth=linwid)
                xd, yd = exterior.xy[:, 0], exterior.xy[:, 1]
                xmin, xmax = min(xmin, np.min(xd)), max(xmax, np.max(xd))
                ymin, ymax = min(ymin, np.min(yd)), max(ymax, np.max(yd))

            for interior in self.boundary_polylines.interiors:
                ax.plot(*interior.xy.T, color=out_col, linewidth=linwid)

        # set plot limits
        xybuf = 6e-3 * (xmax - xmin)
        ax.set_xlim(xmin - xybuf, xmax + xybuf)
        ax.set_ylim(ymin - xybuf, ymax + xybuf)

        if title is not None:
            ax.set_title(title)

        return ax

    def _cbar_extend(self, calc_data, vmin, vmax):
        if calc_data is None:
            return "neither"
        extend_min = calc_data.min() < vmin if vmin is not None else False
        extend_max = calc_data.max() > vmax if vmax is not None else False
        if extend_min and extend_max:
            extend = "both"
        elif extend_min:
            extend = "min"
        elif extend_max:
            extend = "max"
        else:
            extend = "neither"
        return extend

    def _create_tri_only_element_table(self, data=None, geometry=None):
        """Convert quad/tri mesh to pure tri-mesh"""
        if geometry is None:
            geometry = self

        ec = geometry.element_coordinates
        if geometry.is_tri_only:
            return np.vstack(geometry.element_table), ec, data

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

    def _get_boundary_polylines_uncategorized(self):
        """Construct closed polylines for all boundary faces"""
        boundary_faces = self._get_boundary_faces()
        face_remains = boundary_faces.copy()
        polylines = []
        while face_remains.shape[0] > 1:
            n0 = face_remains[:, 0]
            n1 = face_remains[:, 1]
            polyline = [n0[0], n1[0]]
            index_to_delete = [0]
            count = 0
            end_points = face_remains[0, 1]
            while True:
                next_point_index = np.where(n0 == end_points)
                if next_point_index[0].size != 0:
                    polyline.append(face_remains[next_point_index[0][0], 1])
                    index_to_delete.append(next_point_index[0][0])
                    end_points = polyline[-1]
                count += 1
                if count > face_remains.shape[0] or polyline[0] == end_points:
                    break

            face_remains = np.delete(face_remains, index_to_delete, axis=0)
            polylines.append(polyline)
        return polylines

    def _get_boundary_polylines(self):
        """Get boundary polylines and categorize as inner or outer by
        assessing the signed area
        """
        polylines = self._get_boundary_polylines_uncategorized()

        poly_lines_int = []
        poly_lines_ext = []
        Polyline = namedtuple("Polyline", ["n_nodes", "nodes", "xy", "area"])

        for polyline in polylines:
            xy = self.geometry2d.node_coordinates[polyline, :2]
            area = (
                np.dot(xy[:, 1], np.roll(xy[:, 0], 1))
                - np.dot(xy[:, 0], np.roll(xy[:, 1], 1))
            ) * 0.5
            poly_line = np.asarray(polyline)
            xy = self.geometry2d.node_coordinates[poly_line, 0:2]
            poly = Polyline(len(polyline), poly_line, xy, area)
            if area > 0:
                poly_lines_ext.append(poly)
            else:
                poly_lines_int.append(poly)

        BoundaryPolylines = namedtuple(
            "BoundaryPolylines",
            ["n_exteriors", "exteriors", "n_interiors", "interiors"],
        )
        n_ext = len(poly_lines_ext)
        n_int = len(poly_lines_int)
        return BoundaryPolylines(n_ext, poly_lines_ext, n_int, poly_lines_int)

    def _get_boundary_faces(self):
        """Construct list of faces"""
        element_table = self.geometry2d.element_table

        all_faces = []
        for el in element_table:
            ele = [*el, el[0]]
            for j in range(len(el)):
                all_faces.append(ele[j : j + 2])

        all_faces = np.asarray(all_faces)

        all_faces_sorted = np.sort(all_faces, axis=1)
        _, uf_id, face_counts = np.unique(
            all_faces_sorted, axis=0, return_index=True, return_counts=True
        )

        # boundary faces are those appearing only once
        bnd_face_id = face_counts == 1
        return all_faces[uf_id[bnd_face_id]]


class _UnstructuredFile(_UnstructuredGeometry):
    """
    _UnstructuredFile based on _UnstructuredGeometry and base class for Mesh and Dfsu
    knows dotnet file, items and timesteps and reads file header
    """

    show_progress = False

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
            self._type == DfsuFileType.DfsuVerticalProfileSigmaZ
            or self._type == DfsuFileType.Dfsu3DSigmaZ
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
        self._filename = None
        self._source = None
        self._deletevalue = None

        self._n_timesteps = None
        self._start_time = None
        self._timestep_in_seconds = None

        self._n_items = None
        self._items = None
        self._dtype = np.float64

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
        self._type = None  # DfsuFileType.Mesh

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
        self._type = DfsuFileType(dfs.DfsuFileType)
        self._deletevalue = dfs.DeleteValueFloat

        # geometry
        self._set_nodes_from_source(dfs)
        self._set_elements_from_source(dfs)

        if not self.is_2d:
            self._n_layers = dfs.NumberOfLayers
            self._n_sigma = dfs.NumberOfSigmaLayers

        # items
        self._n_items = len(dfs.ItemInfo)
        self._items = _get_item_info(dfs.ItemInfo, list(range(self._n_items)))

        # time
        self._start_time = dfs.StartDateTime
        self._n_timesteps = dfs.NumberOfTimeSteps
        self._timestep_in_seconds = dfs.TimeStepInSeconds

        dfs.Close()

    def _set_nodes_from_source(self, source):
        xn = source.X
        yn = source.Y
        zn = source.Z
        self._nc = np.column_stack([xn, yn, zn])
        self._codes = np.array(list(source.Code))
        self._n_nodes = source.NumberOfNodes
        self._node_ids = source.NodeIds - 1

    def _set_elements_from_source(self, source):
        self._n_elements = source.NumberOfElements
        self._element_table_mikecore = source.ElementTable
        self._element_table = None  # self._element_table_dotnet
        self._element_ids = source.ElementIds - 1


class Dfsu(_UnstructuredFile, EquidistantTimeSeries):
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
        self._filename = str(filename)
        self._read_header(self._filename)
        self._dtype = dtype

        # show progress bar for large files
        # if self._type == UnstructuredType.Mesh:
        #    tot_size = self.n_elements
        # else:
        #    tot_size = self.n_elements * self.n_timesteps * self.n_items
        # if tot_size > 1e6:
        #    self.show_progress = True

    @property
    def deletevalue(self):
        """File delete value"""
        return self._deletevalue

    @property
    def n_items(self):
        """Number of items"""
        return self._n_items

    @property
    def items(self):
        """List of items"""
        return self._items

    @property
    def start_time(self):
        """File start time"""
        return self._start_time

    @property
    def n_timesteps(self):
        """Number of time steps"""
        return self._n_timesteps

    @property
    def timestep(self):
        """Time step size in seconds"""
        return self._timestep_in_seconds

    @property
    def end_time(self):
        """File end time"""
        return self.start_time + timedelta(
            seconds=((self.n_timesteps - 1) * self.timestep)
        )

    def read(self, items=None, time_steps=None, elements=None):
        """
        Read data from a dfsu file

        Parameters
        ---------
        items: list[int] or list[str], optional
            Read only selected items, by number (0-based), or by name
        time_steps: str, int or list[int], optional
            Read only selected time_steps
        elements: list[int], optional
            Read only selected element ids

        Returns
        -------
        Dataset
            A Dataset with data dimensions [t,elements]

        Examples
        --------
        >>> dfsu.read()
        <mikeio.Dataset>
        Dimensions: (9, 884)
        Time: 1985-08-06 07:00:00 - 1985-08-07 03:00:00
        Items:
        0:  Surface elevation <Surface Elevation> (meter)
        1:  U velocity <u velocity component> (meter per sec)
        2:  V velocity <v velocity component> (meter per sec)
        3:  Current speed <Current Speed> (meter per sec)
        >>> dfsu.read(time_steps="1985-08-06 12:00,1985-08-07 00:00")
        <mikeio.Dataset>
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
        # TODO: add more checks that this is actually still the same file
        # (could have been replaced in the meantime)

        item_numbers = _valid_item_numbers(dfs.ItemInfo, items)
        items = _get_item_info(dfs.ItemInfo, item_numbers)
        n_items = len(item_numbers)

        self._n_timesteps = dfs.NumberOfTimeSteps
        time_steps = _valid_timesteps(dfs, time_steps)

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

        for i in trange(len(time_steps), disable=not self.show_progress):
            it = time_steps[i]
            for item in range(n_items):

                itemdata = dfs.ReadItemTimeStep(item_numbers[item] + 1, it)

                src = itemdata.Data

                # d = to_numpy(src)
                d = src

                d[d == deletevalue] = np.nan

                if elements is not None:
                    if item == 0 and item0_is_node_based:
                        d = d[node_ids]
                    else:
                        d = d[elements]

                data_list[item][i, :] = d

            t_seconds[i] = itemdata.Time

        time = pd.to_datetime(t_seconds, unit="s", origin=self.start_time)

        dfs.Close()
        return Dataset(data_list, time, items)

    def extract_track(self, track, items=None, method="nearest"):
        """
        Extract track data from a dfsu file

        Parameters
        ---------
        track: pandas.DataFrame
            with DatetimeIndex and (x, y) of track points as first two columns
            x,y coordinates must be in same coordinate system as dfsu
        track: str
            filename of csv or dfs0 file containing t,x,y
        items: list[int] or list[str], optional
            Extract only selected items, by number (0-based), or by name
        method: str, optional
            Spatial interpolation method ('nearest' or 'inverse_distance')
            default='nearest'

        Returns
        -------
        Dataset
            A dataset with data dimension t
            The first two items will be x- and y- coordinates of track

        Examples
        --------
        >>> ds = dfsu.extract_track(times, xy, items=['u','v'])

        >>> ds = dfsu.extract_track('track_file.dfs0')

        >>> ds = dfsu.extract_track('track_file.csv', items=0)
        """

        dfs = DfsuFile.Open(self._filename)

        item_numbers = _valid_item_numbers(dfs.ItemInfo, items)
        items = _get_item_info(dfs.ItemInfo, item_numbers)
        n_items = len(item_numbers)

        self._n_timesteps = dfs.NumberOfTimeSteps
        time_steps = _valid_timesteps(dfs, time_steps=None)

        deletevalue = self.deletevalue

        if isinstance(track, str):
            filename = track
            if os.path.exists(filename):
                _, ext = os.path.splitext(filename)
                if ext == ".dfs0":
                    df = Dfs0(filename).to_dataframe()
                elif ext == ".csv":
                    df = pd.read_csv(filename, index_col=0, parse_dates=True)
                else:
                    raise ValueError(f"{ext} files not supported (dfs0, csv)")

                times = df.index
                coords = df.iloc[:, 0:2].to_numpy(copy=True)
            else:
                raise ValueError(f"{filename} does not exist")
        elif isinstance(track, Dataset):
            times = track.time
            coords = np.zeros(shape=(len(times), 2))
            coords[:, 0] = track.data[0].copy()
            coords[:, 1] = track.data[1].copy()
        else:
            assert isinstance(track, pd.DataFrame)
            times = track.index
            coords = track.iloc[:, 0:2].to_numpy(copy=True)

        assert isinstance(
            times, pd.DatetimeIndex
        ), "The index must be a pandas.DatetimeIndex"
        assert (
            times.is_monotonic_increasing
        ), "The time index must be monotonic increasing. Consider df.sort_index() before passing to extract_track()."

        data_list = []
        data_list.append(coords[:, 0])  # longitude
        data_list.append(coords[:, 1])  # latitude
        for item in range(n_items):
            # Initialize an empty data block
            data = np.empty(shape=(len(times)), dtype=self._dtype)
            data[:] = np.nan
            data_list.append(data)

        if self.is_geo:
            lon = coords[:, 0]
            lon[lon < -180] = lon[lon < -180] + 360
            lon[lon >= 180] = lon[lon >= 180] - 360
            coords[:, 0] = lon

        # track end (relative to dfsu)
        t_rel = (times - self.end_time).total_seconds()
        # largest idx for which (times - self.end_time)<=0
        tmp = np.where(t_rel <= 0)[0]
        if len(tmp) == 0:
            raise ValueError("No time overlap! Track ends before dfsu starts!")
        i_end = tmp[-1]

        # track time relative to dfsu start
        t_rel = (times - self.start_time).total_seconds()
        tmp = np.where(t_rel >= 0)[0]
        if len(tmp) == 0:
            raise ValueError("No time overlap! Track starts after dfsu ends!")
        i_start = tmp[0]  # smallest idx for which t_rel>=0

        dfsu_step = int(np.floor(t_rel[i_start] / self.timestep))  # first step

        # spatial interpolation
        n_pts = 1 if method == "nearest" else 5
        elem_ids, weights = self.get_2d_interpolant(
            coords[i_start : (i_end + 1)], n_nearest=n_pts
        )

        # initialize dfsu data arrays
        d1 = np.ndarray(shape=(n_items, self.n_elements), dtype=self._dtype)
        d2 = np.ndarray(shape=(n_items, self.n_elements), dtype=self._dtype)
        t1 = 0.0
        t2 = 0.0

        # very first dfsu time step
        step = time_steps[dfsu_step]
        for item in range(n_items):
            itemdata = dfs.ReadItemTimeStep(item_numbers[item] + 1, step)
            t2 = itemdata.Time - 1e-10
            # d = to_numpy(itemdata.Data)
            d = itemdata.Data
            d[d == deletevalue] = np.nan
            d2[item, :] = d

        def is_EOF(step):
            return step >= self.n_timesteps

        # loop over track points
        for i_interp, i in enumerate(
            trange(i_start, i_end + 1, disable=not self.show_progress)
        ):
            t_rel[i]  # time of point relative to dfsu start

            read_next = t_rel[i] > t2

            while (read_next == True) and (not is_EOF(dfsu_step + 1)):
                dfsu_step = dfsu_step + 1

                # swap new to old
                d1, d2 = d2, d1
                t1, t2 = t2, t1

                step = time_steps[dfsu_step]
                for item in range(n_items):
                    itemdata = dfs.ReadItemTimeStep(item_numbers[item] + 1, step)
                    t2 = itemdata.Time
                    # d = to_numpy(itemdata.Data)
                    d = itemdata.Data
                    d[d == deletevalue] = np.nan
                    d2[item, :] = d

                read_next = t_rel[i] > t2

            if (read_next == True) and (is_EOF(dfsu_step)):
                # cannot read next - no more timesteps in dfsu file
                continue

            w = (t_rel[i] - t1) / self.timestep  # time-weight
            eid = elem_ids[i_interp]
            if np.any(eid > 0):
                dati = (1 - w) * np.dot(d1[:, eid], weights[i_interp])
                dati = dati + w * np.dot(d2[:, eid], weights[i_interp])
            else:
                dati = np.empty(shape=n_items, dtype=self._dtype)
                dati[:] = np.nan

            for item in range(n_items):
                data_list[item + 2][i] = dati[item]

        dfs.Close()

        items_out = []
        if self.is_geo:
            items_out.append(ItemInfo("Longitude"))
            items_out.append(ItemInfo("Latitude"))
        else:
            items_out.append(ItemInfo("x"))
            items_out.append(ItemInfo("y"))
        for item in items:
            items_out.append(item)

        return Dataset(data_list, times, items_out)

    def extract_surface_elevation_from_3d(
        self, filename=None, time_steps=None, n_nearest=4
    ):
        """
        Extract surface elevation from a 3d dfsu file (based on zn)
        to a new 2d dfsu file with a surface elevation item.

        Parameters
        ---------
        filename: str
            Output file name
        time_steps: str, int or list[int], optional
            Extract only selected time_steps
        n_nearest: int, optional
            number of points for spatial interpolation (inverse_distance), default=4

        Examples
        --------
        >>> dfsu.extract_surface_elevation_from_3d('ex_surf.dfsu', time_steps='2018-1-1,2018-2-1')
        """
        # validate input
        assert (
            self._type == DfsuFileType.Dfsu3DSigma
            or self._type == DfsuFileType.Dfsu3DSigmaZ
        )
        assert n_nearest > 0
        time_steps = _valid_timesteps(self._source, time_steps)

        # make 2d nodes-to-elements interpolator
        top_el = self.top_elements
        geom = self.elements_to_geometry(top_el, node_layers="top")
        xye = geom.element_coordinates[:, 0:2]
        xyn = geom.node_coordinates[:, 0:2]
        tree2d = cKDTree(xyn)
        dist, node_ids = tree2d.query(xye, k=n_nearest)
        if n_nearest == 1:
            weights = None
        else:
            weights = get_idw_interpolant(dist)

        # read zn from 3d file and interpolate to element centers
        ds = self.read(items=0, time_steps=time_steps)  # read only zn
        node_ids_surf, _ = self._get_nodes_and_table_for_elements(
            top_el, node_layers="top"
        )
        zn_surf = ds.data[0][:, node_ids_surf]  # surface
        surf2d = interp2d(zn_surf, node_ids, weights)

        # create output
        items = [ItemInfo(EUMType.Surface_Elevation)]
        ds2 = Dataset([surf2d], ds.time, items)
        if filename is None:
            return ds2
        else:
            title = "Surface extracted from 3D file"
            self.write(filename, ds2, elements=top_el, title=title)

    def write_header(
        self,
        filename,
        start_time=None,
        dt=None,
        items=None,
        elements=None,
        title=None,
    ):
        """Write the header of a new dfsu file (for writing huge files)

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
        filename = str(filename)

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
            items = [ItemInfo(f"Item {i + 1}") for i in range(n_items)]

        if title is None:
            title = ""

        file_start_time = start_time

        # spatial subset
        if elements is None:
            geometry = self
        else:
            geometry = self.elements_to_geometry(elements)
            if (not self.is_2d) and (geometry._type == DfsuFileType.Dfsu2D):
                # redo extraction as 2d:
                # print("will redo extraction in 2d!")
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
        if geometry._type is None:  # == DfsuFileType.Mesh:
            # create dfs2d from mesh
            dfsu_filetype = DfsuFileType.Dfsu2D
        else:
            #    # TODO: if subset is slice...
            dfsu_filetype = geometry._type.value

        if dfsu_filetype != DfsuFileType.Dfsu2D:
            if items[0].name != "Z coordinate":
                raise Exception("First item must be z coordinates of the nodes!")

        xn = geometry.node_coordinates[:, 0]
        yn = geometry.node_coordinates[:, 1]

        # zn have to be Single precision??
        zn = geometry.node_coordinates[:, 2]

        # TODO verify this
        # elem_table = geometry.element_table
        elem_table = []
        for j in range(geometry.n_elements):
            elem_nodes = geometry.element_table[j]
            elem_nodes = [nd + 1 for nd in elem_nodes]
            elem_table.append(np.array(elem_nodes))
        elem_table = elem_table

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
            for i in trange(n_time_steps, disable=not self.show_progress):
                for item in range(n_items):
                    d = data[item][i, :]
                    d[np.isnan(d)] = deletevalue
                    darray = d
                    self._dfs.WriteItemTimeStepNext(0, darray.astype(np.float32))
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
        for i in trange(n_time_steps, disable=not self.show_progress):
            for item in range(n_items):
                d = data[item][i, :]
                d[np.isnan(d)] = deletevalue
                # darray = to_dotnet_float_array(d)
                darray = d.astype(np.float32)
                self._dfs.WriteItemTimeStepNext(0, darray.astype(np.float32))

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
            # make sure element table has been constructured
            _ = self.element_table
            geometry = self
        else:
            geometry = self.geometry2d

        Mesh._geometry_to_mesh(outfilename, geometry)

    def to_dfs2(
        self,
        x0: float,
        y0: float,
        dx: float,
        dy: float,
        nx: int = 20,
        ny: int = 20,
        rotation: float = 0,
        epsg: typing.Optional[int] = None,
        interpolation_method: str = "nearest",
        filename: typing.Optional[typing.Union[str, pathlib.Path]] = None,
    ):
        """Export Dfsu to Dfs2 file.

        Export Dfsu file to a Dfs2 file with a regular 2D grid.

        Parameters
        ----------
        x0 : float
            X-coordinate of the bottom left corner of the 2D grid,
            must be in the same coordinate system as the parent Dfsu file.
        y0 : float
            Y-coordinate of the bottom left corner of the 2D grid,
            must be in the same coordinate system as the parent Dfsu file.
        dx : float
            Grid resolution in the X direction in the units of CRS defined by `epsg`.
        dy : float
            Grid resolution in the Y direction in the units of CRS defined by `epsg`.
        nx : int, optional
            Grid size in the X direction. By default it is 20.
        ny : int, optional
            Grid size in the Y direction. By default it is 20.
        rotation : float, optional
            Grid clockwise rotation in degrees. Be default it is 0.
        epsg : int, optional
            EPSG identificator of coordinate system
            in which the Dfs2 file will be created.
            If None (default), uses coordinate system of the parent Dfsu file.
        interpolation_method : str, optional
            Interpolation method, by default it is 'nearest'.
        filename : str or pathlib.Path, optional
            Path to dfs2 file to be created.
            If None (default), creates a temporary dfs2 file
            in the system temporary directory.

        Returns
        -------
        Dfs2
            mikeio Dfs2 object pointing to the file located at `filename`.

        """
        # Process 'filename' argument
        if filename is None:
            filename = tempfile.NamedTemporaryFile().name + ".dfs2"
        else:
            if isinstance(filename, str):
                filename = pathlib.Path(filename)

            if isinstance(filename, pathlib.Path):
                filename = filename.resolve()
                if not filename.suffix == ".dfs2":
                    raise ValueError(
                        f"'filename' must point to a dfs2 file, "
                        f"not to '{filename.suffix}'"
                    )
            else:
                raise TypeError(
                    f"invalid type in '{type(filename)}' for the 'filename' argument, "
                    f"must be string or pathlib.Path"
                )

        # Define 2D grid in 'epsg' projection
        grid = Grid2D(
            bbox=[
                x0,
                y0,
                x0 + dx * nx,
                y0 + dy * ny,
            ],
            shape=(nx, ny),
        )
        # TODO - create rotated grid
        if rotation != 0:
            raise NotImplementedError(
                "'rotation' argument is currently not supported, "
                "grid is assumed to have its y-axis pointing at True North"
            )

        # Determine Dfsu projection
        # Convert X/Y points from Dfsu to 'epsg' projection
        # TODO - infer CRS and transform between Dfsu and Dfs2 coordinate sytems
        if epsg is not None:
            raise NotImplementedError(
                "'epsg' argument is currently not supported, "
                "coordinate system is taken from the parent Dfsu file"
            )

        # Interpolate Dfsu items to 2D grid using scipy.interpolate.griddata
        # TODO - interpolate between Dfs2 and Dfsu grids, taking into account
        # TODO - interpolation method, CRS, and grid rotation
        if interpolation_method != "nearest":
            raise NotImplementedError(
                "'interpolation_method' argument is currently not supported, "
                "interpolation is performed using nearest neighborhood method"
            )
        elem_ids, weights = self.get_2d_interpolant(
            xy=grid.xy,
            n_nearest=1,
            extrapolate=False,
            p=2,
            radius=None,
        )
        dataset = self.read(items=None, time_steps=None, elements=None)
        interpolated_dataset = self.interp2d(
            dataset,
            elem_ids=elem_ids,
            weights=weights,
            shape=(grid.ny, grid.nx),
        )

        interpolated_dataset = interpolated_dataset.flipud()

        # Write interpolated data to 'filename'
        dfs2 = Dfs2()
        dfs2.write(
            filename=str(filename),
            data=interpolated_dataset,
            start_time=dataset.time[0].to_pydatetime(),
            dt=dataset.timestep,
            items=self.items,
            dx=grid.dx,
            dy=grid.dy,
            coordinate=[
                self.projection_string,  # projection
                grid.x0,  # origin_x
                grid.y0,  # orign_y
                0,  # grid orientation - TODO account for 'rotation' argument
            ],
            title=None,  # TODO - infer it from parent Dfsu
        )

        # Return reference to the created Dfs2 file
        return Dfs2(filename=str(filename))


class Mesh(_UnstructuredFile):
    """
    The Mesh class is initialized with a mesh or a dfsu file.

    Parameters
    ---------
    filename: str
        dfsu or mesh filename

    Examples
    --------

    >>> msh = Mesh("../tests/testdata/odense_rough.mesh")
    >>> msh
    Number of elements: 654
    Number of nodes: 399
    Projection: UTM-33

    """

    def __init__(self, filename):
        super().__init__()
        self._filename = filename
        self._read_header(filename)
        self._n_timesteps = None
        self._n_items = None
        self._n_layers = None
        self._n_sigma = None
        self._type = None  # DfsuFileType.Mesh

    @property
    def zn(self):
        """Static bathymetry values (depth) at nodes"""
        return self.node_coordinates[:, 2]

    @zn.setter
    def zn(self, v):
        if len(v) != self.n_nodes:
            raise ValueError(f"zn must have length of nodes ({self.n_nodes})")
        self._nc[:, 2] = v
        self._ec = None

    def set_z(self, z):
        """Deprecated: use msh.zn = values instead

        Change the depth by setting the z value of each node

        Parameters
        ----------
        z : np.array(float)
            new z value at each node
        """
        warnings.warn(
            "Mesh.set_z is deprecated, please use msh.zn = values instead.",
            FutureWarning,
        )
        self.zn = z

    def set_codes(self, codes):
        """Deprecated: use msh.codes = values instead

        Change the code values of the nodes

        Parameters
        ----------
        codes : list(int)
            code of each node
        """
        warnings.warn(
            "Mesh.set_codes is deprecated, please use msh.codes = values instead.",
            FutureWarning,
        )
        self.codes = codes

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
            if hasattr(self._source, "EumQuantity"):
                quantity = self._source.EumQuantity
            else:
                quantity = eumQuantity.Create(EUMType.Bathymetry, self._source.ZUnit)
            elem_table = self._source.ElementTable
        else:
            geometry = self.elements_to_geometry(elements)
            quantity = eumQuantity.Create(EUMType.Bathymetry, EUMUnit.meter)
            elem_table = geometry._element_table_to_mikecore()

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
        builder.SetElements(geometry._element_table_to_mikecore())
        builder.SetProjection(geometry.projection_string)
        quantity = eumQuantity.Create(EUMType.Bathymetry, EUMUnit.meter)
        builder.SetEumQuantity(quantity)
        newMesh = builder.CreateMesh()
        newMesh.Write(outfilename)
