from typing import Sequence, Union
import warnings
import numpy as np
from collections import namedtuple
from scipy.spatial import cKDTree
from mikecore.DfsuFile import DfsuFileType
from mikecore.eum import eumQuantity
from mikecore.MeshBuilder import MeshBuilder

from ..eum import EUMType, EUMUnit
from .geometry import _Geometry, BoundingBox, GeometryPoint2D, GeometryPoint3D
from .grid_geometry import Grid2D
from ..interpolation import get_idw_interpolant, interp2d
from ..custom_exceptions import InvalidGeometry
from .FM_utils import _get_node_centered_data, _to_polygons, _plot_map


class GeometryFMPointSpectrum(_Geometry):
    def __init__(self) -> None:
        super().__init__()
        self.n_nodes = 0
        self.n_elements = 0
        self._type = DfsuFileType.DfsuSpectral0D
        self.is_layered = False
        self.is_2d = False
        self.is_spectral = True

    @property
    def type_name(self):
        """Type name: DfsuSpectral0D"""
        return self._type.name

    def __repr__(self):
        return "Flexible Mesh Point Geometry (empty)"

    @property
    def ndim(self):
        return 0


class _GeometryFMPlotter:
    def __init__(self, geometry: "GeometryFM") -> None:
        self.g = geometry

    def __call__(self, ax=None, figsize=None, **kwargs):
        ax = self._get_ax(ax, figsize)
        return self._plot_FM_map(ax, **kwargs)

    def contour(self, ax=None, figsize=None, **kwargs):
        ax = self._get_ax(ax, figsize)
        kwargs["plot_type"] = "contour"
        return self._plot_FM_map(ax, **kwargs)

    def contourf(self, ax=None, figsize=None, **kwargs):
        ax = self._get_ax(ax, figsize)
        kwargs["plot_type"] = "contourf"
        return self._plot_FM_map(ax, **kwargs)

    @staticmethod
    def _get_ax(ax=None, figsize=None):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        return ax

    def _plot_FM_map(self, ax, **kwargs):

        if "title" not in kwargs:
            kwargs["title"] = "Bathymetry"

        g = self.g._geometry2d

        return _plot_map(
            node_coordinates=g.node_coordinates,
            element_table=g.element_table,
            element_coordinates=g.element_coordinates,
            boundary_polylines=g.boundary_polylines,
            is_geo=g.is_geo,
            z=None,
            ax=ax,
            **kwargs,
        )

    def mesh(self, title="Mesh", figsize=None, ax=None):
        from matplotlib.collections import PatchCollection

        ax = self._get_ax(ax=ax, figsize=figsize)
        ax.set_aspect(self._plot_aspect())

        patches = _to_polygons(
            self.g._geometry2d.node_coordinates, self.g._geometry2d.element_table
        )
        fig_obj = PatchCollection(
            patches, edgecolor="0.6", facecolor="none", linewidths=0.3
        )
        ax.add_collection(fig_obj)
        self.outline(ax=ax)
        ax.set_title(title)
        return ax

    def outline(self, title="Outline", figsize=None, ax=None):
        ax = self._get_ax(ax=ax, figsize=figsize)
        ax.set_aspect(self._plot_aspect())

        linwid = 1.2
        out_col = "0.4"
        for exterior in self.g.boundary_polylines.exteriors:
            ax.plot(*exterior.xy.T, color=out_col, linewidth=linwid)
        for interior in self.g.boundary_polylines.interiors:
            ax.plot(*interior.xy.T, color=out_col, linewidth=linwid)
        ax.set_title(title)
        return ax

    def boundary_nodes(self, boundary_names=None, figsize=None, ax=None):
        """
        Plot mesh boundary nodes and their codes
        """
        import matplotlib.pyplot as plt

        ax = self._get_ax(ax=ax, figsize=figsize)
        ax.set_aspect(self._plot_aspect())

        nc = self.g.node_coordinates
        c = self.g.codes
        valid_codes = list(set(self.g.codes))
        boundary_codes = [code for code in valid_codes if code > 0]

        if boundary_names is not None:
            if len(boundary_codes) != len(boundary_names):
                raise Exception(
                    f"Number of boundary names ({len(boundary_names)}) inconsistent with number of boundaries ({len(self.g.boundary_codes)})"
                )
            user_defined_labels = dict(zip(boundary_codes, boundary_names))

        for code in boundary_codes:
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

    def _plot_aspect(self):
        if self.g.is_geo:
            mean_lat = np.mean(self.g.node_coordinates[:, 1])
            return 1.0 / np.cos(np.pi * mean_lat / 180)
        else:
            return "equal"


class GeometryFM(_Geometry):
    def __init__(
        self,
        node_coordinates,
        element_table,
        codes=None,
        projection=None,
        dfsu_type=None,
        element_ids=None,
        node_ids=None,
    ) -> None:
        super().__init__()

        if node_coordinates is None:
            raise ValueError("node_coordinates must be provided")
        if element_table is None:
            raise ValueError("element_table must be provided")

        self._type = None  # None: mesh, 0: 2d-dfsu, 4:dfsu3dsigma, ...

        self._nc = None
        self._ec = None
        self._codes = None
        self._element_ids = None
        self._node_ids = None
        self._element_table = None
        self._n_axis = 1
        self._n_layers = None

        self._tree2d = None
        self._boundary_polylines = None
        self._geom2d = None

        self._set_nodes(
            node_coordinates=node_coordinates,
            codes=codes,
            node_ids=node_ids,
            projection_string=projection,
        )

        self._set_elements(
            element_table=element_table,
            element_ids=element_ids,
            dfsu_type=dfsu_type,
        )

        self.plot = _GeometryFMPlotter(self)

    def __repr__(self):
        out = []
        out.append("Flexible Mesh Geometry")
        if self.n_nodes:
            out.append(f"Number of nodes: {self.n_nodes}")
        if self.n_elements:
            out.append(f"Number of elements: {self.n_elements}")
        if self._n_layers:
            out.append(f"Number of layers: {self._n_layers}")
        if self._projstr:
            out.append(f"Projection: {self.projection_string}")
        return str.join("\n", out)

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

        for i, e in enumerate(element_table):

            if not isinstance(e, np.ndarray):
                e = np.array(e)
                element_table[i] = e
            if e.max() > (self.node_ids.max()):
                raise ValueError(
                    f"Element table has node # {e.max()}. Max node id: {self.node_ids.max()}"
                )

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

    def _reindex(self):
        new_node_ids = np.arange(self.n_nodes)
        new_element_ids = np.arange(self.n_elements)
        node_dict = dict(zip(self._node_ids, new_node_ids))
        for j in range(self.n_elements):
            elem_nodes = self._element_table[j]
            new_elem_nodes = []
            for idx in elem_nodes:
                new_elem_nodes.append(node_dict[idx])
            self._element_table[j] = new_elem_nodes

        self._node_ids = new_node_ids
        self._element_ids = new_element_ids

    @property
    def node_coordinates(self):
        """Coordinates (x,y,z) of all nodes"""
        return self._nc

    @property
    def n_nodes(self) -> int:
        """Number of nodes"""
        return None if self._node_ids is None else len(self._node_ids)

    @property
    def node_ids(self):
        return self._node_ids

    @property
    def n_elements(self) -> int:
        """Number of elements"""
        return None if self._element_ids is None else len(self._element_ids)

    @property
    def element_ids(self):
        return self._element_ids

    @property
    def element_table(self):
        """Element to node connectivity"""
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
    def is_2d(self) -> bool:
        """Type is either mesh or Dfsu2D (2 horizontal dimensions)"""
        return self._type in (
            DfsuFileType.Dfsu2D,
            DfsuFileType.DfsuSpectral2D,
            None,
        )

    @property
    def is_layered(self) -> bool:
        """Type is layered dfsu (3d, vertical profile or vertical column)"""
        return self._type in (
            DfsuFileType.DfsuVerticalColumn,
            DfsuFileType.DfsuVerticalProfileSigma,
            DfsuFileType.DfsuVerticalProfileSigmaZ,
            DfsuFileType.Dfsu3DSigma,
            DfsuFileType.Dfsu3DSigmaZ,
        )

    @property
    def is_spectral(self) -> bool:
        """Type is spectral dfsu (point, line or area spectrum)"""
        return self._type in (
            DfsuFileType.DfsuSpectral0D,
            DfsuFileType.DfsuSpectral1D,
            DfsuFileType.DfsuSpectral2D,
        )

    @property
    def is_tri_only(self) -> bool:
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

        if self.is_layered:
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
        xy = self._geometry2d.element_coordinates[:, :2]
        self._tree2d = cKDTree(xy)

    def _find_n_nearest_2d_elements(self, x, y=None, n=1):
        if n > self._geometry2d.n_elements:
            raise ValueError(
                f"Cannot find {n} nearest! Number of 2D elements: {self._geometry2d.n_elements}"
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

    def get_overset_grid(self, dx=None, dy=None, shape=None, buffer=None) -> Grid2D:
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
        nc = self._geometry2d.node_coordinates
        bbox = Grid2D.xy_to_bbox(nc, buffer=buffer)
        return Grid2D(bbox=bbox, dx=dx, dy=dy, shape=shape, projection=self.projection)

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

    @property
    def codes(self):
        """Node codes of all nodes (0=water, 1=land, 2...=open boundaries)"""
        return self._codes

    @property
    def boundary_polylines(self):
        """Lists of closed polylines defining domain outline"""
        if self._boundary_polylines is None:
            self._boundary_polylines = self._get_boundary_polylines()
        return self._boundary_polylines

    def contains(self, points) -> Sequence[bool]:
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
            xy = self._geometry2d.node_coordinates[polyline, :2]
            area = (
                np.dot(xy[:, 1], np.roll(xy[:, 0], 1))
                - np.dot(xy[:, 0], np.roll(xy[:, 1], 1))
            ) * 0.5
            poly_line = np.asarray(polyline)
            xy = self._geometry2d.node_coordinates[poly_line, 0:2]
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
        element_table = self._geometry2d.element_table

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

    def isel(self, idx=None, axis="elements", simplify=True):

        if np.isscalar(idx) == 1 and simplify:
            coords = self.element_coordinates[idx].flatten()

            if self.is_layered:
                return GeometryPoint3D(*coords)
            else:
                return GeometryPoint2D(coords[0], coords[1])
        else:
            return self.elements_to_geometry(elements=idx, node_layers=None)

    def elements_to_geometry(
        self, elements, node_layers="all"
    ) -> Union["GeometryFM", "GeometryFMLayered"]:
        """export a selection of elements to new flexible file geometry

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
        elements = [elements] if np.isscalar(elements) else elements
        elements = np.sort(elements)  # make sure elements are sorted!

        # create new geometry
        new_type = self._type
        if self.is_layered:
            layers_used = self.layer_ids[elements]
            unique_layer_ids = np.unique(layers_used)
            n_layers = len(unique_layer_ids)
            if (
                self._type == DfsuFileType.Dfsu3DSigma
                or self._type == DfsuFileType.Dfsu3DSigmaZ
            ) and n_layers == 1:
                new_type = DfsuFileType.Dfsu2D

            if n_layers == 1 and node_layers in ("all", None):
                node_layers = "bottom"

        # extract information for selected elements
        node_ids, elem_tbl = self._get_nodes_and_table_for_elements(
            elements, node_layers=node_layers
        )
        node_coords = self.node_coordinates[node_ids]
        codes = self.codes[node_ids]

        if self.is_layered and (new_type != DfsuFileType.Dfsu2D):
            GeometryClass = GeometryFMLayered
        else:
            GeometryClass = GeometryFM

        geom = GeometryClass(
            node_coordinates=node_coords,
            codes=codes,
            node_ids=node_ids,
            projection=self.projection_string,
            element_table=elem_tbl,
            element_ids=self.element_ids[elements],
        )
        geom._reindex()

        geom._type = self._type  #
        if self.is_layered:
            if new_type == DfsuFileType.Dfsu2D:
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
        geometry = self._geometry2d
        nc = geometry.node_coordinates
        ec = geometry.element_coordinates
        elem_table = geometry.element_table
        return _get_node_centered_data(nc, elem_table, ec, data, extrapolate)

    @property
    def _geometry2d(self):
        """The 2d geometry for a 3d object"""
        if self._n_layers is None:
            return self
        if self._geom2d is None:
            self._geom2d = self.to_2d_geometry()
        return self._geom2d

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

    def to_mesh(self, outfilename):

        builder = MeshBuilder()

        nc = self.node_coordinates
        builder.SetNodes(nc[:, 0], nc[:, 1], nc[:, 2], self.codes)
        # builder.SetNodeIds(self.node_ids+1)
        # builder.SetElementIds(self.elements+1)
        element_table_MZ = [np.asarray(row) + 1 for row in self.element_table]
        builder.SetElements(element_table_MZ)
        builder.SetProjection(self.projection_string)
        quantity = eumQuantity.Create(EUMType.Bathymetry, EUMUnit.meter)
        builder.SetEumQuantity(quantity)
        newMesh = builder.CreateMesh()
        newMesh.Write(outfilename)


# class GeometryFMHorizontal(GeometryFM):
#     pass


class GeometryFMLayered(GeometryFM):
    def __init__(
        self,
        node_coordinates=None,
        element_table=None,
        codes=None,
        projection=None,
        dfsu_type=None,
        element_ids=None,
        node_ids=None,
        n_layers=None,
        n_sigma=None,
    ) -> None:
        super().__init__(
            node_coordinates=node_coordinates,
            element_table=element_table,
            codes=codes,
            projection=projection,
            dfsu_type=dfsu_type,
            element_ids=element_ids,
            node_ids=node_ids,
        )
        self._top_elems = None
        self._n_layers_column = None
        self._bot_elems = None
        self._n_layers = n_layers
        self._n_sigma = n_sigma

        self._geom2d = None
        self._e2_e3_table = None
        self._2d_ids = None
        self._layer_ids = None

    @property
    def geometry2d(self):
        """The 2d geometry for a 3d object"""
        return self._geometry2d

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
        geom = GeometryFM(
            node_coordinates=node_coords,
            codes=codes,
            node_ids=node_ids,
            projection=self.projection_string,
            element_table=elem_tbl,
            element_ids=self.element_ids[elem_ids],
        )

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

    @property
    def e2_e3_table(self):
        """The 2d-to-3d element connectivity table for a 3d object"""
        if self.n_layers is None:
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
        if self.n_layers is None:
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
        if self.n_layers is None:
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
        if self.n_layers is None:
            return None
        return self.n_layers - self.n_sigma_layers

    @property
    def top_elements(self):
        """List of 3d element ids of surface layer"""
        if self.n_layers is None:
            print("Object has no layers: cannot find top_elements")
            return None
        elif self._top_elems is None:
            # note: if subset of elements is selected then this cannot be done!

            # TODO: check 0-based, 1-based...
            self._top_elems = self._findTopLayerElements(self.element_table)
        return self._top_elems

    @staticmethod
    def _findTopLayerElements(elementTable):
        """
        Find element indices (zero based) of the elements being the upper-most element
        in its column.
        Each column is identified by matching node id numbers. For 3D elements the
        last half of the node numbers of the bottom element must match the first half
        of the node numbers in the top element. For 2D vertical elements the order of
        the node numbers in the bottom element (last half number of nodes) are reversed
        compared to those in the top element (first half number of nodes).
        To find the number of elements in each column, assuming the result
        is stored in res:
        For the first column it is res[0]+1.
        For the i'th column, it is res[i]-res[i-1].
        :returns: A list of element indices of top layer elements
        """

        topLayerElments = []

        # Find top layer elements by matching the number numers of the last half of elmt i
        # with the first half of element i+1.
        # Elements always start from the bottom, and the element of one columne are following
        # each other in the element table.
        for i in range(len(elementTable) - 1):
            elmt1 = elementTable[i]
            elmt2 = elementTable[i + 1]

            if len(elmt1) != len(elmt2):
                # elements with different number of nodes can not be on top of each other,
                # so elmt2 must be another column, and elmt1 must be a top element
                topLayerElments.append(i)
                continue

            if len(elmt1) % 2 != 0:
                raise Exception(
                    "In a layered mesh, each element must have an even number of elements (element index "
                    + i
                    + ")"
                )

            # Number of nodes in a 2D element
            elmt2DSize = len(elmt1) // 2

            for j in range(elmt2DSize):
                if elmt2DSize > 2:
                    if elmt1[j + elmt2DSize] != elmt2[j]:
                        # At least one node number did not match
                        # so elmt2 must be another column, and elmt1 must be a top element
                        topLayerElments.append(i)
                        break
                else:
                    # for 2D vertical profiles the nodes in the element on the
                    # top is in reverse order of those in the bottom.
                    if elmt1[j + elmt2DSize] != elmt2[(elmt2DSize - 1) - j]:
                        # At least one node number did not match
                        # so elmt2 must be another column, and elmt1 must be a top element
                        topLayerElments.append(i)
                        break

        # The last element will always be a top layer element
        topLayerElments.append(len(elementTable) - 1)

        return np.array(topLayerElments, dtype=np.int32)

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
        if isinstance(layer, str):
            if layer in ("surface", "top"):
                return self.top_elements
            elif layer in ("bottom"):
                return self.bottom_elements
            else:
                raise ValueError(
                    f"layer '{layer}' not recognized ('top', 'bottom' or integer)"
                )

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


# class GeometryFMSpectral(GeometryFM):
#     # TODO: add specialized classes: FrequencySpectrum, DirectionalSpectrum
#     def __init__(
#         self,
#         frequencies=None,
#         directions=None,
#         node_coordinates=None,
#         element_table=None,
#         codes=None,
#         projection_string=None,
#         dfsu_type=None,
#     ) -> None:
#         super().__init__(
#             node_coordinates=node_coordinates,
#             element_table=element_table,
#             codes=codes,
#             projection_string=projection_string,
#             dfsu_type=dfsu_type,
#         )
#         self._frequencies = frequencies
#         self._directions = directions
#         self._n_axis = 0 if (self.n_elements == 0) else 1
#         self._n_axis = (
#             self._n_axis + int(self.n_frequencies > 0) + int(self.n_directions > 0)
#         )

#     @property
#     def n_frequencies(self):
#         """Number of frequencies"""
#         return 0 if self.frequencies is None else len(self.frequencies)

#     @property
#     def frequencies(self):
#         """Frequency axis"""
#         return self._frequencies

#     @property
#     def n_directions(self):
#         """Number of directions"""
#         return 0 if self.directions is None else len(self.directions)

#     @property
#     def directions(self):
#         """Directional axis"""
#         return self._directions
