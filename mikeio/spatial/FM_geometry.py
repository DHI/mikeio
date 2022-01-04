import numpy as np
from collections import namedtuple
from mikecore.DfsuFile import DfsuFileType
from geometry import _Geometry, BoundingBox


class GeometryFM(_Geometry):
    def __init__(
        self,
        node_coordinates=None,
        element_table=None,
        codes=None,
        projection_string=None,
        dfsu_type=None,
    ) -> None:
        super().__init__()

        self._type = None  # None: mesh, 0: 2d-dfsu, 4:dfsu3dsigma, ...

        self._nc = None
        self._ec = None
        self._codes = None
        self._element_ids = None
        self._node_ids = None
        self._element_table = None
        self._n_axis = 1

        self._tree2d = None

        if node_coordinates is not None:
            self._set_nodes(
                node_coordinates=node_coordinates,
                codes=codes,
                node_ids=None,
                projection_string=projection_string,
            )

        if element_table is not None:
            self._set_elements(
                element_table=element_table,
                element_ids=None,
                dfsu_type=dfsu_type,
            )

    # should projection string still be here?
    def _set_nodes(
        self, node_coordinates, codes=None, node_ids=None, projection_string=None
    ):
        self._nc = np.asarray(node_coordinates)
        if codes is None:
            codes = np.zeros(len(node_coordinates), dtype=int)
        self._codes = np.asarray(codes)
        if node_ids is None:
            node_ids = np.arange(len(codes))
        self._node_ids = np.asarray(node_ids)
        self._projstr = "LONG/LAT" if projection_string is None else projection_string

    def _set_elements(self, element_table, element_ids=None, dfsu_type=None):
        self._element_table = element_table
        if element_ids is None:
            element_ids = np.arange(len(element_table))
        self._element_ids = np.asarray(element_ids)

        if dfsu_type is None:
            # guess type
            if self.max_nodes_per_element < 5:
                dfsu_type = DfsuFileType.Dfsu2D
            else:
                dfsu_type = DfsuFileType.Dfsu3DSigma
        self._type = dfsu_type

    @property
    def node_coordinates(self):
        """Coordinates (x,y,z) of all nodes"""
        return self._nc

    @property
    def n_nodes(self):
        """Number of nodes"""
        return len(self._node_ids)

    @property
    def n_elements(self):
        """Number of elements"""
        return len(self._element_ids)

    @property
    def element_table(self):
        """Element to node connectivity"""
        if (self._element_table is None) and (self._element_table_mikecore is not None):
            self._element_table = self._get_element_table_from_mikecore()
        return self._element_table

    # cache this?
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
    def type_name(self):
        """Type name, e.g. Mesh, Dfsu2D"""
        return self._type.name if self._type else "Mesh"

    @property
    def is_tri_only(self):
        """Does the mesh consist of triangles only?"""
        return self.max_nodes_per_element == 3 or self.max_nodes_per_element == 6

    @property
    def element_coordinates(self):
        """Center coordinates of each element"""
        if self._ec is None:
            self._ec = self._calc_element_coordinates()
        return self._ec

    def _calc_element_coordinates(self, elements=None, zn=None):
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

    @property
    def boundary_polylines(self):
        """Lists of closed polylines defining domain outline"""
        if self._boundary_polylines is None:
            self._boundary_polylines = self._get_boundary_polylines()
        return self._boundary_polylines

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


class GeometryFMHorizontal(GeometryFM):
    pass


class GeometryFMLayered(GeometryFM):
    def __init__(
        self,
        node_coordinates=None,
        element_table=None,
        codes=None,
        projection_string=None,
        dfsu_type=None,
    ) -> None:
        super().__init__(
            node_coordinates=node_coordinates,
            element_table=element_table,
            codes=codes,
            projection_string=projection_string,
            dfsu_type=dfsu_type,
        )
        self._top_elems = None
        self._n_layers_column = None
        self._bot_elems = None
        self._n_layers = None
        self._n_sigma = None

        self._geom2d = None
        self._e2_e3_table = None
        self._2d_ids = None
        self._layer_ids = None

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

        Returns
        -------
        np.array
            x,y,z of each element
        """
        return self._calc_element_coordinates(elements, zn)

    # TODO: add methods for extracting layers etc


class GeometryFMSpectral(GeometryFM):
    # TODO: add specialized classes: FrequencySpectrum, DirectionalSpectrum
    def __init__(
        self,
        frequencies=None,
        directions=None,
        node_coordinates=None,
        element_table=None,
        codes=None,
        projection_string=None,
        dfsu_type=None,
    ) -> None:
        super().__init__(
            node_coordinates=node_coordinates,
            element_table=element_table,
            codes=codes,
            projection_string=projection_string,
            dfsu_type=dfsu_type,
        )
        self._frequencies = frequencies
        self._directions = directions
        self._n_axis = 0 if (self.n_elements == 0) else 1
        self._n_axis = (
            self._n_axis + int(self.n_frequencies > 0) + int(self.n_directions > 0)
        )

    @property
    def n_frequencies(self):
        """Number of frequencies"""
        return 0 if self.frequencies is None else len(self.frequencies)

    @property
    def frequencies(self):
        """Frequency axis"""
        return self._frequencies

    @property
    def n_directions(self):
        """Number of directions"""
        return 0 if self.directions is None else len(self.directions)

    @property
    def directions(self):
        """Directional axis"""
        return self._directions
