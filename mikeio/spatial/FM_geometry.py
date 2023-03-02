import warnings
from collections import namedtuple
from functools import cached_property
from typing import Collection, Sequence, Union

import numpy as np
from mikecore.DfsuFile import DfsuFileType
from mikecore.eum import eumQuantity
from mikecore.MeshBuilder import MeshBuilder
from scipy.spatial import cKDTree

from ..eum import EUMType, EUMUnit
from ..exceptions import InvalidGeometry, OutsideModelDomainError
from ..interpolation import get_idw_interpolant, interp2d
from .FM_utils import (
    _get_node_centered_data,
    _plot_map,
    _plot_vertical_profile,
    BoundaryPolylines,
    _set_xy_label_by_projection,  # TODO remove
    _to_polygons,  # TODO remove
)
from .geometry import GeometryPoint2D, GeometryPoint3D, _Geometry
from .grid_geometry import Grid2D
from .utils import _relative_cumulative_distance, xy_to_bbox


class GeometryFMPointSpectrum(_Geometry):
    def __init__(self, frequencies=None, directions=None, x=None, y=None) -> None:
        super().__init__()
        self.n_nodes = 0
        self.n_elements = 0
        self.is_2d = False
        self.is_spectral = True

        self._frequencies = frequencies
        self._directions = directions
        self.x = x
        self.y = y

    @property
    def type_name(self):
        """Type name: DfsuSpectral0D"""
        return self._type.name  # TODO there is no self._type??

    def __repr__(self):
        txt = f"Point Spectrum Geometry(frequency:{self.n_frequencies}, direction:{self.n_directions}"
        if self.x is not None:
            txt = txt + f", x:{self.x:.5f}, y:{self.y:.5f}"
        return txt + ")"

    @property
    def ndim(self):
        # TODO: 0, 1 or 2 ?
        return 0

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


class _GeometryFMPlotter:
    """Plot GeometryFM

    Examples
    --------
    >>> ds = mikeio.read("HD2D.dfsu")
    >>> g = ds.geometry
    >>> g.plot()          # bathymetry (as patches)
    >>> g.plot.contour()  # bathymetry contours
    >>> g.plot.contourf() # filled bathymetry contours
    >>> g.plot.mesh()     # mesh only
    >>> g.plot.outline()  # domain outline only
    >>> g.plot.boundary_nodes()
    """

    def __init__(self, geometry: "GeometryFM") -> None:
        self.g = geometry

    def __call__(self, ax=None, figsize=None, **kwargs):
        """Plot bathymetry as coloured patches"""
        ax = self._get_ax(ax, figsize)
        kwargs["plot_type"] = kwargs.get("plot_type") or "patch"
        return self._plot_FM_map(ax, **kwargs)

    def contour(self, ax=None, figsize=None, **kwargs):
        """Plot bathymetry as contour lines"""
        ax = self._get_ax(ax, figsize)
        kwargs["plot_type"] = "contour"
        return self._plot_FM_map(ax, **kwargs)

    def contourf(self, ax=None, figsize=None, **kwargs):
        """Plot bathymetry as filled contours"""
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

        plot_type = kwargs.pop("plot_type")

        g = self.g._geometry2d

        return _plot_map(
            node_coordinates=g.node_coordinates,
            element_table=g.element_table,
            element_coordinates=g.element_coordinates,
            boundary_polylines=g.boundary_polylines,
            plot_type=plot_type,
            projection=g.projection,
            z=None,
            ax=ax,
            **kwargs,
        )

    def mesh(self, title="Mesh", figsize=None, ax=None):
        """Plot mesh only"""

        # TODO this must be a duplicate, delegate

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
        ax = self._set_plot_limits(ax)
        _set_xy_label_by_projection(ax, self.g.projection)
        return ax

    def outline(self, title="Outline", figsize=None, ax=None):
        """Plot domain outline (using the boundary_polylines property)"""

        # TODO this must be a duplicate, delegate
        ax = self._get_ax(ax=ax, figsize=figsize)
        ax.set_aspect(self._plot_aspect())

        linwid = 1.2
        out_col = "0.4"
        for exterior in self.g.boundary_polylines.exteriors:
            ax.plot(*exterior.xy.T, color=out_col, linewidth=linwid)
        for interior in self.g.boundary_polylines.interiors:
            ax.plot(*interior.xy.T, color=out_col, linewidth=linwid)
        if title is not None:
            ax.set_title(title)
        ax = self._set_plot_limits(ax)
        return ax

    def boundary_nodes(self, boundary_names=None, figsize=None, ax=None):
        """Plot mesh boundary nodes and their code values"""
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
        ax = self._set_plot_limits(ax)
        return ax

    def _set_plot_limits(self, ax):
        # TODO this must be a duplicate, delegate
        bbox = xy_to_bbox(self.g.node_coordinates)
        xybuf = 6e-3 * (bbox.right - bbox.left)
        ax.set_xlim(bbox.left - xybuf, bbox.right + xybuf)
        ax.set_ylim(bbox.bottom - xybuf, bbox.top + xybuf)
        return ax

    def _plot_aspect(self):
        # TODO this must be a duplicate, delegate
        if self.g.is_geo:
            mean_lat = np.mean(self.g.node_coordinates[:, 1])
            return 1.0 / np.cos(np.pi * mean_lat / 180)
        else:
            return "equal"


class _GeometryFMVerticalProfilePlotter:
    def __init__(self, geometry: "GeometryFM") -> None:
        self.g = geometry

    def __call__(self, ax=None, figsize=None, **kwargs):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        x = self.g.node_coordinates[:, 0]
        y = self.g.node_coordinates[:, 1]
        ax.plot(x, y, **kwargs)
        return ax

    def mesh(self, title="Mesh", edge_color="0.5", **kwargs):

        v = np.full_like(self.g.element_coordinates[:, 0], np.nan)
        return _plot_vertical_profile(
            node_coordinates=self.g.node_coordinates,
            element_table=self.g.element_table,
            values=v,
            is_geo=self.g.is_geo,
            title=title,
            add_colorbar=False,
            edge_color=edge_color,
            cmin=0.0,
            cmax=1.0,
            **kwargs,
        )


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
        validate=True,
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
            validate=validate,
        )

        self.plot = _GeometryFMPlotter(self)

        # try:
        #     import numba

        #     self._point_in_polygon = numba.njit(_point_in_polygon)
        # except ModuleNotFoundError:
        # self._point_in_polygon = _point_in_polygon

    def _repr_txt(self, layer_txt=None):
        out = []
        out.append("Flexible Mesh Geometry: " + self.type_name)
        if self.n_nodes:
            out.append(f"number of nodes: {self.n_nodes}")
        if self.n_elements:
            out.append(f"number of elements: {self.n_elements}")
            if layer_txt is not None:
                out.append(layer_txt)
        if self._projstr:
            out.append(f"projection: {self.projection_string}")
        return str.join("\n", out)

    def __repr__(self):
        return self._repr_txt()

    def __str__(self) -> str:
        gtxt = f"{self.type_name}"
        if self.is_layered:
            n_z_layers = "no" if self.n_z_layers is None else self.n_z_layers
            gtxt += f" ({self.n_elements} elements, {self.n_sigma_layers} sigma-layers, {n_z_layers} z-layers)"
        else:
            gtxt += f" ({self.n_elements} elements, {self.n_nodes} nodes)"
        return gtxt

    @staticmethod
    def _point_in_polygon(xn: np.array, yn: np.array, xp: float, yp: float) -> bool:
        """Check for each side in the polygon that the point is on the correct side"""

        for j in range(len(xn) - 1):
            if (yn[j + 1] - yn[j]) * (xp - xn[j]) + (-xn[j + 1] + xn[j]) * (
                yp - yn[j]
            ) > 0:
                return False
            if (yn[0] - yn[-1]) * (xp - xn[-1]) + (-xn[0] + xn[-1]) * (yp - yn[-1]) > 0:
                return False
        return True

    @staticmethod
    def _area_is_bbox(area) -> bool:
        is_bbox = False
        if area is not None:
            if not np.isscalar(area):
                area = np.array(area)
                if (area.ndim == 1) & (len(area) == 4):
                    if np.all(np.isreal(area)):
                        is_bbox = True
        return is_bbox

    @staticmethod
    def _area_is_polygon(area) -> bool:
        if area is None:
            return False
        if np.isscalar(area):
            return False
        if not np.all(np.isreal(area)):
            return False
        polygon = np.array(area)
        if polygon.ndim > 2:
            return False

        if polygon.ndim == 1:
            if len(polygon) <= 5:
                return False
            if len(polygon) % 2 != 0:
                return False

        if polygon.ndim == 2:
            if polygon.shape[0] < 3:
                return False
            if polygon.shape[1] != 2:
                return False

        return True

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

    def _set_elements(
        self, element_table, element_ids=None, dfsu_type=None, validate=True
    ):

        if validate:
            max_node_id = self.node_ids.max()
            for i, e in enumerate(element_table):
                # TODO: avoid looping through all elements (could be +1e6)!
                if not isinstance(e, np.ndarray):
                    e = np.asarray(e)
                    element_table[i] = e

                # NOTE: this check "e.max()" takes the most of the time when constructing a new FM_geometry
                if e.max() > max_node_id:
                    raise ValueError(
                        f"Element table has node # {e.max()}. Max node id: {max_node_id}"
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
        for eid in range(self.n_elements):
            elem_nodes = self._element_table[eid]
            new_elem_nodes = np.zeros_like(elem_nodes)
            for jn, idx in enumerate(elem_nodes):
                new_elem_nodes[jn] = node_dict[idx]
            self._element_table[eid] = new_elem_nodes

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
    def ndim(self) -> int:
        if self.is_layered:
            return 3
        else:
            return 2

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
        >>> g = dfs.geometry
        >>> id = g.find_nearest_elements(3, 4)
        >>> ids = g.find_nearest_elements([3, 8], [4, 6])
        >>> ids = g.find_nearest_elements(xy)
        >>> ids = g.find_nearest_elements(3, 4, n_nearest=4)
        >>> ids, d = g.find_nearest_elements(xy, return_distances=True)

        >>> ids = g.find_nearest_elements(3, 4, z=-3)
        >>> ids = g.find_nearest_elements(3, 4, layer=4)
        >>> ids = g.find_nearest_elements(xyz)
        >>> ids = g.find_nearest_elements(xyz, n_nearest=3)

        See Also
        --------
        find_index : find element indicies for points or an area
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
        self,
        xy,
        n_nearest: int = 5,
        extrapolate: bool = False,
        p: int = 2,
        radius: float = None,
    ):
        """IDW interpolant for list of coordinates

        Parameters
        ----------
        xy : array-like
            x,y coordinates of new points
        n_nearest : int, optional
            number of nearest elements used for IDW, by default 5
        extrapolate : bool, optional
            allow extrapolation, by default False
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

    def _find_element_2d(self, coords: np.array):

        points_outside = []

        coords = np.atleast_2d(coords)
        nc = self._geometry2d.node_coordinates

        few_nearest, _ = self._find_n_nearest_2d_elements(
            coords, n=min(self._geometry2d.n_elements, 2)
        )
        ids = np.atleast_2d(few_nearest)[:, 0]  # first guess

        for k in range(len(ids)):
            # step 1: is nearest element = element containing point?
            nodes = self._geometry2d.element_table[ids[k]]
            element_found = self._point_in_polygon(
                nc[nodes, 0], nc[nodes, 1], coords[k, 0], coords[k, 1]
            )

            # step 2: if not, then try second nearest point
            if not element_found and self._geometry2d.n_elements > 1:
                candidate = few_nearest[k, 1]
                assert np.isscalar(candidate)
                nodes = self._geometry2d.element_table[candidate]
                element_found = self._point_in_polygon(
                    nc[nodes, 0], nc[nodes, 1], coords[k, 0], coords[k, 1]
                )
                ids[k] = few_nearest[k, 1]

            # step 3: if not, then try with *many* more points
            if not element_found and self._geometry2d.n_elements > 1:
                many_nearest, _ = self._find_n_nearest_2d_elements(
                    coords[k, :],
                    n=min(self._geometry2d.n_elements, 10),  # TODO is 10 enough?
                )
                for p in many_nearest[2:]:  # we have already tried the two first above
                    nodes = self._geometry2d.element_table[p]
                    element_found = self._point_in_polygon(
                        nc[nodes, 0], nc[nodes, 1], coords[k, 0], coords[k, 1]
                    )
                    if element_found:
                        ids[k] = p
                        break

            if not element_found:
                points_outside.append(k)

        if len(points_outside) > 0:
            raise OutsideModelDomainError(
                x=coords[points_outside, 0],
                y=coords[points_outside, 1],
                indices=points_outside,
            )

        return ids

    def _find_single_element_2d(self, x: float, y: float) -> int:

        nc = self._geometry2d.node_coordinates

        few_nearest, _ = self._find_n_nearest_2d_elements(
            x=x, y=y, n=min(self.n_elements, 10)
        )

        for idx in few_nearest:
            nodes = self._geometry2d.element_table[idx]
            element_found = self._point_in_polygon(nc[nodes, 0], nc[nodes, 1], x, y)

            if element_found:
                return idx

        raise OutsideModelDomainError(x=x, y=y)

    def get_overset_grid(
        self, dx=None, dy=None, nx=None, ny=None, buffer=None
    ) -> Grid2D:
        """get a 2d grid that covers the domain by specifying spacing or shape

        Parameters
        ----------
        dx : float or (float, float), optional
            grid resolution in x-direction (or in x- and y-direction)
        dy : float, optional
            grid resolution in y-direction
        nx : int, optional
            number of points in x-direction, by default None,
            (the value will be inferred)
        ny : int, optional
            number of points in y-direction, by default None,
            (the value will be inferred)
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
        bbox = xy_to_bbox(nc, buffer=buffer)
        return Grid2D(bbox=bbox, dx=dx, dy=dy, nx=nx, ny=ny, projection=self.projection)

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

    @codes.setter
    def codes(self, v):
        if len(v) != self.n_nodes:
            raise ValueError(f"codes must have length of nodes ({self.n_nodes})")
        self._codes = np.array(v, dtype=np.int32)

    @property
    def boundary_polylines(self) -> BoundaryPolylines:
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

    def __contains__(self, pt) -> bool:
        return self.contains(pt)

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

    def _get_boundary_polylines(self) -> BoundaryPolylines:
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

    def isel(
        self, idx: Collection[int], keepdims=False, **kwargs
    ) -> Union["GeometryFM", "GeometryFM3D", GeometryPoint2D, GeometryPoint3D]:
        """export a selection of elements to a new geometry

        Typically not called directly, but by Dataset/DataArray's
        isel() or sel() methods.

        Parameters
        ----------
        idx : collection(int)
            collection of element indicies
        keepdims : bool, optional
            Should the original Geometry type be kept (keepdims=True)
            or should it be reduced e.g. to a GeometryPoint2D if possible
            (keepdims=False), by default False

        Returns
        -------
        Geometry
            geometry subset

        See Also
        --------
        find_index : find element indicies for points or an area
        """

        if self._type == DfsuFileType.DfsuSpectral1D:
            return self._nodes_to_geometry(nodes=idx)
        else:
            return self.elements_to_geometry(
                elements=idx, node_layers=None, keepdims=keepdims
            )

    def find_index(self, x=None, y=None, coords=None, area=None) -> np.ndarray:
        """Find a *set* of element indicies for a number of points or within an area.

        The returned indices returned are the unique, unordered set of element indices that contain the points or area.

        This method will return elements *containing* the argument
        points/area, which is not necessarily the same as the nearest.

        Typically not called directly, but by Dataset/DataArray's
        sel() method.

        Parameters
        ----------
        x: float or array(float)
            X coordinate(s) (easting or longitude)
        y: float or array(float)
            Y coordinate(s) (northing or latitude)
        coords : np.array(float,float), optional
            As an alternative to specifying x, and y individually,
            the argument coords can be used instead.
            (x,y)-coordinates of points to be found,
            by default None
        area : (float, float, float, float), optional
            Bounding box of coordinates (left lower and right upper)
            to be selected, by default None

        Returns
        -------
        np.array
            indicies of containing elements

        Raises
        ------
        ValueError
            if any point is outside the domain

        Examples
        --------
        >>> g = dfs.geometry
        >>> id = dfs.find_index(x=3.1, y=4.3)

        See Also
        --------
        isel : get subset geometry for specific indicies
        find_nearest_elements : find nearest instead of containing elements
        """
        if (coords is not None) or (x is not None) or (y is not None):
            if area is not None:
                raise ValueError(
                    "Coordinates and area cannot be provided at the same time!"
                )
            if coords is not None:
                coords = np.atleast_2d(coords)
                xy = coords[:, :2]
            else:
                xy = np.vstack((x, y)).T
            idx = self._find_element_2d(coords=xy)
            return idx
        elif area is not None:
            return self._elements_in_area(area)
        else:
            raise ValueError("Provide either coordinates or area")

    def _find_elem3d_from_elem2d(self, elem2d, z):
        """Find 3d element ids from 2d element ids and z-values"""

        # TODO: coordinate with _find_3d_from_2d_points()

        elem2d = [elem2d] if np.isscalar(elem2d) else elem2d
        elem2d = np.asarray(elem2d)
        z_vec = np.full(elem2d.shape, fill_value=z) if np.isscalar(z) else z
        elem3d = np.full_like(elem2d, fill_value=-1)
        for j, e2 in enumerate(elem2d):
            idx_3d = np.hstack(self.e2_e3_table[e2])
            elem3d[j] = idx_3d[self._z_idx_in_column(idx_3d, z_vec[j])]

            # z_col = self.element_coordinates[idx_3d, 2]
            # elem3d[j] = (np.abs(z_col - z_vec[j])).argmin()  # nearest
        return elem3d

    def _z_idx_in_column(self, e3_col, z):
        dz = self._dz[e3_col]
        z_col = self.element_coordinates[e3_col, 2]
        z_face = np.append(z_col - dz / 2, z_col[-1] + dz[-1] / 2)
        if z < z_face[0] or z > z_face[-1]:
            xy = tuple(self.element_coordinates[e3_col[0], :2])
            raise ValueError(
                f"z value '{z}' is outside water column [{z_face[0]},{z_face[-1]}] in point x,y={xy}"
            )
        idx = np.searchsorted(z_face, z) - 1
        return idx

    def _elements_in_area(self, area):
        """Find element ids of elements inside area"""
        idx = self._2d_elements_in_area(area)
        if self.is_layered and len(idx) > 0:
            idx = np.hstack(self.e2_e3_table[idx])
        return idx

    def _2d_elements_in_area(self, area):
        """Find 2d element ids of elements inside area"""
        if self._area_is_bbox(area):
            x0, y0, x1, y1 = area
            xc = self._geometry2d.element_coordinates[:, 0]
            yc = self._geometry2d.element_coordinates[:, 1]
            mask = (xc >= x0) & (xc <= x1) & (yc >= y0) & (yc <= y1)
        elif self._area_is_polygon(area):
            polygon = np.array(area)
            xy = self._geometry2d.element_coordinates[:, :2]
            mask = self._inside_polygon(polygon, xy)
        else:
            raise ValueError("'area' must be bbox [x0,y0,x1,y1] or polygon")

        return np.where(mask)[0]

    def _nodes_to_geometry(self, nodes) -> "GeometryFM":
        """export a selection of nodes to new flexible file geometry

        Note: takes only the elements for which all nodes are selected

        Parameters
        ----------
        nodes : list(int)
            list of node ids

        Returns
        -------
        UnstructuredGeometry
            which can be used for further extraction or saved to file
        """
        assert not self.is_layered, "not supported for layered data"

        nodes = np.atleast_1d(nodes)
        if len(nodes) == 1:
            xy = self.node_coordinates[nodes[0], :2]
            return GeometryPoint2D(xy[0], xy[1])

        elements = []
        for j, el_nodes in enumerate(self.element_table):
            if np.all(np.isin(el_nodes, nodes)):
                elements.append(j)

        assert len(elements) > 0, "no elements found"
        elements = np.sort(elements)  # make sure elements are sorted!

        node_ids, elem_tbl = self._get_nodes_and_table_for_elements(elements)
        node_coords = self.node_coordinates[node_ids]
        codes = self.codes[node_ids]

        geom = GeometryFM(
            node_coordinates=node_coords,
            codes=codes,
            node_ids=node_ids,
            projection=self.projection_string,
            element_table=elem_tbl,
            element_ids=self.element_ids[elements],
        )
        geom._reindex()
        geom._type = self._type
        return geom

    def elements_to_geometry(
        self, elements: Union[int, Collection[int]], node_layers="all", keepdims=False
    ) -> Union["GeometryFM", "GeometryFM3D", GeometryPoint3D, GeometryPoint2D]:
        """export a selection of elements to new flexible file geometry

        Parameters
        ----------
        elements : int or Collection[int]
            collection of element ids
        node_layers : str, optional
            for 3d files either 'top', 'bottom' layer nodes
            or 'all' can be selected, by default 'all'
        keepdims: bool, optional
            keep original geometry type for single points

        Returns
        -------
        UnstructuredGeometry
            which can be used for further extraction or saved to file
        """
        if np.isscalar(elements):
            elements = [elements]
        else:
            elements = list(elements)
        if len(elements) == 1 and not keepdims:
            coords = self.element_coordinates[elements.pop(), :]
            if self.is_layered:
                return GeometryPoint3D(*coords, projection=self.projection)
            else:
                return GeometryPoint2D(coords[0], coords[1], projection=self.projection)

        elements = np.sort(
            elements
        )  # make sure elements are sorted! # TODO is this necessary?

        # create new geometry
        new_type = self._type
        if self.is_layered:
            elements = list(elements)
            layers_used = self.layer_ids[elements]
            unique_layer_ids = np.unique(layers_used)
            n_layers = len(unique_layer_ids)

            if n_layers > 1:
                elem_bot = self.get_layer_elements("bottom")
                if np.all(np.in1d(elements, elem_bot)):
                    n_layers = 1

            if (
                self._type == DfsuFileType.Dfsu3DSigma
                or self._type == DfsuFileType.Dfsu3DSigmaZ
            ) and n_layers == 1:
                new_type = DfsuFileType.Dfsu2D

            if n_layers == 1 and node_layers in ("all", None):
                node_layers = "bottom"

        # extract information for selected elements
        if self.is_layered and n_layers == 1:
            geom2d = self._geometry2d
            elem2d = self.elem2d_ids[elements]
            node_ids, elem_tbl = geom2d._get_nodes_and_table_for_elements(elem2d)
            node_coords = geom2d.node_coordinates[node_ids]
            codes = geom2d.codes[node_ids]
            elem_ids = self.element_ids[elem2d]
        else:
            node_ids, elem_tbl = self._get_nodes_and_table_for_elements(
                elements, node_layers=node_layers
            )
            node_coords = self.node_coordinates[node_ids]
            codes = self.codes[node_ids]
            elem_ids = self.element_ids[elements]

        if self.is_layered and (new_type != DfsuFileType.Dfsu2D):
            if n_layers == len(elem_tbl):
                GeometryClass = GeometryFMVerticalColumn
            else:
                GeometryClass = self.__class__
        else:
            GeometryClass = GeometryFM

        geom = GeometryClass(
            node_coordinates=node_coords,
            codes=codes,
            node_ids=node_ids,
            projection=self.projection_string,
            element_table=elem_tbl,
            element_ids=elem_ids,
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
                lowest_sigma = self.n_layers - self.n_sigma_layers
                geom._n_sigma = sum(unique_layer_ids >= lowest_sigma)

                # If source is sigma-z but output only has sigma layers
                # then change type accordingly
                if (
                    self._type == DfsuFileType.DfsuVerticalProfileSigmaZ
                    or self._type == DfsuFileType.Dfsu3DSigmaZ
                ) and n_layers == geom._n_sigma:
                    # TODO fix this
                    geom._type = DfsuFileType.Dfsu3DSigma

                geom._top_elems = geom._findTopLayerElements(geom.element_table)

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
        elem_tbl = np.empty(len(elements), dtype=np.dtype("O"))
        if (node_layers is None) or (node_layers == "all") or self.is_2d:
            for j, eid in enumerate(elements):
                elem_tbl[j] = np.asarray(self.element_table[eid])

        else:
            # 3D => 2D
            if (node_layers != "bottom") and (node_layers != "top"):
                raise Exception("node_layers must be either all, bottom or top")
            for j, eid in enumerate(elements):
                elem_nodes = np.asarray(self.element_table[eid])
                nn = len(elem_nodes)
                halfn = int(nn / 2)
                if node_layers == "bottom":
                    elem_nodes = elem_nodes[:halfn]
                if node_layers == "top":
                    elem_nodes = elem_nodes[halfn:]
                elem_tbl[j] = elem_nodes

        nodes = np.unique(np.hstack(elem_tbl))
        return nodes, elem_tbl

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
        from shapely.geometry import MultiPolygon, Polygon

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
        """Export geometry to new mesh file

        Parameters
        ----------
        outfilename : str
            path to file to be written
        """
        builder = MeshBuilder()

        geom2d = self._geometry2d

        nc = geom2d.node_coordinates
        builder.SetNodes(nc[:, 0], nc[:, 1], nc[:, 2], geom2d.codes)
        # builder.SetNodeIds(geom2d.node_ids+1)
        # builder.SetElementIds(geom2d.elements+1)
        element_table_MZ = [np.asarray(row) + 1 for row in geom2d.element_table]
        builder.SetElements(element_table_MZ)
        builder.SetProjection(geom2d.projection_string)
        quantity = eumQuantity.Create(EUMType.Bathymetry, EUMUnit.meter)
        builder.SetEumQuantity(quantity)
        newMesh = builder.CreateMesh()
        newMesh.Write(outfilename)


# class GeometryFMHorizontal(GeometryFM):
#     pass


class _GeometryFMLayered(GeometryFM):
    def __init__(
        self,
        node_coordinates,
        element_table,
        codes=None,
        projection=None,
        dfsu_type=None,
        element_ids=None,
        node_ids=None,
        n_layers: int = 1,  # at least 1 layer
        n_sigma=None,
        validate=True,
    ) -> None:
        super().__init__(
            node_coordinates=node_coordinates,
            element_table=element_table,
            codes=codes,
            projection=projection,
            dfsu_type=dfsu_type,
            element_ids=element_ids,
            node_ids=node_ids,
            validate=validate,
        )
        self._n_layers_column = None
        self._bot_elems = None
        self._n_layers = n_layers
        self._n_sigma = n_sigma

        self._geom2d = None
        self._e2_e3_table = None
        self._2d_ids = None
        self._layer_ids = None
        self.__dz = None

    def __repr__(self):
        details = (
            "sigma only"
            if self.n_z_layers is None
            else f"{self.n_sigma_layers} sigma-layers, max {self.n_z_layers} z-layers"
        )
        layer_txt = f"number of layers: {self._n_layers} ({details})"
        return self._repr_txt(layer_txt=layer_txt)

    @property
    def layer_ids(self):
        """The layer number (0=bottom, 1, 2, ...) for each 3d element"""
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
        return self.n_layers - self.n_sigma_layers

    @cached_property
    def top_elements(self):
        """List of 3d element ids of surface layer"""
        # note: if subset of elements is selected then this cannot be done!

        # fast path if no z-layers
        if self.n_z_layers == 0:
            return np.arange(
                start=self.n_sigma_layers - 1,
                stop=self.n_elements,
                step=self.n_sigma_layers,
            )
        else:
            # slow path
            return self._findTopLayerElements(self.element_table)

    def find_index(self, x=None, y=None, z=None, coords=None, area=None, layers=None):

        if layers is not None:
            idx = self.get_layer_elements(layers)
        else:
            idx = self.element_ids

        if (
            (coords is not None)
            or (x is not None)
            or (y is not None)
            or (z is not None)
        ):
            if area is not None:
                raise ValueError(
                    "Coordinates and area cannot be provided at the same time!"
                )
            if coords is not None:
                coords = np.atleast_2d(coords)
                xy = coords[:, :2]
                z = coords[:, 2] if coords.shape[1] == 3 else None
            else:
                xy = np.vstack((x, y)).T
            idx_2d = self._find_element_2d(coords=xy)
            assert len(idx_2d) == len(xy)
            if z is None:
                idx_3d = np.hstack(self.e2_e3_table[idx_2d])
            else:
                idx_3d = self._find_elem3d_from_elem2d(idx_2d, z)
            idx = np.intersect1d(idx, idx_3d).astype(int)
        elif area is not None:
            idx_area = self._elements_in_area(area)
            idx = np.intersect1d(idx, idx_area)
        elif layers is None:
            raise ValueError(
                "At least one selection argument (x,y,z,coords,area,layers) needs to be provided!"
            )
        return idx

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

    def get_layer_elements(self, layers, layer=None):
        """3d element ids for one (or more) specific layer(s)

        Parameters
        ----------
        layers : int or list(int)
            layer between 0 (bottom) and n_layers-1 (top)
            (can also be negative counting from -1 at the top layer)

        Returns
        -------
        np.array(int)
            element ids
        """
        if layer is not None:
            warnings.warn(
                "layer argument is deprecated, use layers instead",
                FutureWarning,
            )
            layers = layer

        if isinstance(layers, str):
            if layers in ("surface", "top"):
                return self.top_elements
            elif layers in ("bottom"):
                return self.bottom_elements
            else:
                raise ValueError(
                    f"layers '{layers}' not recognized ('top', 'bottom' or integer)"
                )

        if not np.isscalar(layers):
            elem_ids = []
            for layer in layers:
                elem_ids.append(self.get_layer_elements(layer))
            elem_ids = np.concatenate(elem_ids, axis=0)
            return np.sort(elem_ids)

        n_lay = self.n_layers
        if n_lay is None:
            raise InvalidGeometry("Object has no layers: cannot get_layer_elements")

        if layers < (-n_lay) or layers >= n_lay:
            raise Exception(
                f"Layer {layers} not allowed; must be between -{n_lay} and {n_lay-1}"
            )

        if layers < 0:
            layers = layers + n_lay

        return self.element_ids[self.layer_ids == layers]

    @property
    def e2_e3_table(self):
        """The 2d-to-3d element connectivity table for a 3d object"""
        if self._e2_e3_table is None:
            res = self._get_2d_to_3d_association()
            self._e2_e3_table = res[0]
            self._2d_ids = res[1]
            self._layer_ids = res[2]
        return self._e2_e3_table

    @property
    def elem2d_ids(self):
        """The associated 2d element id for each 3d element"""
        if self._2d_ids is None:
            res = self._get_2d_to_3d_association()
            self._e2_e3_table = res[0]
            self._2d_ids = res[1]
            self._layer_ids = res[2]
        return self._2d_ids

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
            col = np.arange(botid[j], topid[j] + 1)

            e2_to_e3.append(col)
            for _ in col:
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
            idx = self.top_elements[elem2d]  # TODO: return whole column instead

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
                idx = self.get_layer_elements(layer)[elem2d]

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

    @property
    def _dz(self):
        """Height of each 3d element (using static zn information)"""
        if self.__dz is None:
            self.__dz = self._calc_dz()
        return self.__dz

    def _calc_dz(self, elements=None, zn=None):
        """Height of 3d elements using static or dynamic zn information"""
        if elements is None:
            element_table = self.element_table
        else:
            element_table = self.element_table[elements]
        n_elements = len(element_table)

        if zn is None:
            if elements is None:
                zn = self.node_coordinates[:, 2]
            else:
                nodes = np.unique(np.hstack(element_table))
                zn = self.node_coordinates[nodes, 2]

        zn_is_2d = len(zn.shape) == 2
        shape = (zn.shape[0], n_elements) if zn_is_2d else (n_elements,)
        dz = np.full(shape=shape, fill_value=np.nan)

        if zn_is_2d:
            # dynamic zn
            for j in range(n_elements):
                nodes = element_table[j]
                halfn = int(len(nodes) / 2)
                z_bot = np.mean(zn[:, nodes[:halfn]], axis=1)
                z_top = np.mean(zn[:, nodes[halfn:]], axis=1)
                dz[:, j] = z_top - z_bot
        else:
            # static zn
            for j in range(n_elements):
                nodes = element_table[j]
                halfn = int(len(nodes) / 2)
                z_bot = np.mean(zn[nodes[:halfn]])
                z_top = np.mean(zn[nodes[halfn:]])
                dz[j] = z_top - z_bot

        return dz

    # TODO: add methods for extracting layers etc


class GeometryFM3D(_GeometryFMLayered):
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
            validate=False,
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


class GeometryFMVerticalProfile(_GeometryFMLayered):
    def __init__(
        self,
        node_coordinates,
        element_table,
        codes=None,
        projection=None,
        dfsu_type=None,
        element_ids=None,
        node_ids=None,
        n_layers: int = 1,
        n_sigma=None,
        validate=True,
    ) -> None:
        super().__init__(
            node_coordinates,
            element_table,
            codes,
            projection,
            dfsu_type,
            element_ids,
            node_ids,
            n_layers,
            n_sigma,
            validate,
        )
        self.plot = _GeometryFMVerticalProfilePlotter(self)
        # self._rel_node_dist = None
        self._rel_elem_dist = None

        # remove inherited but unsupported methods
        self.interp2d = None

    @property
    def boundary_polylines(self):
        # Overides base class
        raise AttributeError(
            "GeometryFMVerticalProfile has no boundary_polylines property"
        )

    # remove unsupported methods from dir to avoid confusion
    def __dir__(self):
        unsupported = ("boundary_polylines", "interp2d")
        keys = sorted(list(super().__dir__()) + list(self.__dict__.keys()))
        return set([d for d in keys if d not in unsupported])

    # @property
    # def relative_node_distance(self):
    #     if self._rel_node_dist is None:
    #         nc = self.node_coordinates
    #         self._rel_node_dist = _relative_cumulative_distance(nc, is_geo=self.is_geo)
    #     return self._rel_node_dist

    @property
    def relative_element_distance(self):
        if self._rel_elem_dist is None:
            ec = self.element_coordinates
            nc0 = self.node_coordinates[0, :2]
            self._rel_elem_dist = _relative_cumulative_distance(
                ec, nc0, is_geo=self.is_geo
            )
        return self._rel_elem_dist

    def get_nearest_relative_distance(self, coords) -> float:
        """For a point near a transect, find the nearest relative distance
        for showing position on transect plot.

        Parameters
        ----------
        coords : [float, float]
            x,y-coordinate of point

        Returns
        -------
        float
            relative distance in meters from start of transect
        """
        xe = self.element_coordinates[:, 0]
        ye = self.element_coordinates[:, 1]
        dd2 = np.square(xe - coords[0]) + np.square(ye - coords[1])
        idx = np.argmin(dd2)
        return self.relative_element_distance[idx]

    def find_index(self, x=None, y=None, z=None, coords=None, layers=None):

        if layers is not None:
            idx = self.get_layer_elements(layers)
        else:
            idx = self.element_ids

        # select in space
        if (
            (coords is not None)
            or (x is not None)
            or (y is not None)
            or (z is not None)
        ):
            if coords is not None:
                coords = np.atleast_2d(coords)
                xy = coords[:, :2]
                z = coords[:, 2] if coords.shape[1] == 3 else None
            else:
                xy = np.vstack((x, y)).T

            idx_2d = self._find_nearest_element_2d(coords=xy)

            if z is None:
                idx_3d = np.hstack(self.e2_e3_table[idx_2d])
            else:
                idx_3d = self._find_elem3d_from_elem2d(idx_2d, z)
            idx = np.intersect1d(idx, idx_3d)

        return idx

    def _find_nearest_element_2d(self, coords):
        ec2d = self.element_coordinates[self.top_elements, :2]
        xe, ye = ec2d[:, 0], ec2d[:, 1]
        coords = np.atleast_2d(coords)
        idx = np.zeros(len(coords), dtype=int)
        for j, pt in enumerate(coords):
            x, y = pt[0:2]
            idx[j] = np.argmin((xe - x) ** 2 + (ye - y) ** 2)
        return idx


class GeometryFMVerticalColumn(GeometryFM3D):
    def calc_ze(self, zn=None):
        if zn is None:
            zn = self.node_coordinates[:, 2]
        return self._calc_z_using_idx(zn, self._idx_e)

    def calc_zf(self, zn=None):
        if zn is None:
            zn = self.node_coordinates[:, 2]
        return self._calc_z_using_idx(zn, self._idx_f)

    def _calc_zee(self, zn=None):
        ze = self.calc_ze(zn)
        zf = self.calc_zf(zn)
        if ze.ndim == 1:
            zee = np.insert(ze, 0, zf[0])
            return np.append(zee, zf[-1])
        else:
            return np.hstack(
                (zf[:, 0].reshape((-1, 1)), ze, zf[:, -1].reshape((-1, 1)))
            )

    def _interp_values(self, zn, data, z):
        """Interpolate to other z values, allow linear extrapolation"""
        from scipy.interpolate import interp1d

        opt = {"kind": "linear", "bounds_error": False, "fill_value": "extrapolate"}

        ze = self.calc_ze(zn)
        dati = np.zeros_like(z)
        if zn.ndim == 1:
            dati = interp1d(ze, data, **opt)(z)
        elif zn.ndim == 2:
            for j in range(zn.shape[0]):
                dati[j, :] = interp1d(ze[j, :], data[j, :], **opt)(z[j, :])
        return dati

    @property
    def _idx_f(self):
        nnodes_half = int(len(self.element_table[0]) / 2)
        n_vfaces = self.n_elements + 1
        idx_f = np.zeros((n_vfaces, nnodes_half), dtype=int)
        idx_e = self._idx_e
        idx_f[: self.n_elements, :] = idx_e[:, :nnodes_half]
        idx_f[self.n_elements, :] = idx_e[-1, nnodes_half:]
        return idx_f

    @property
    def _idx_e(self):
        nnodes_per_elem = len(self.element_table[0])
        idx_e = np.zeros((self.n_elements, nnodes_per_elem), dtype=int)

        for j in range(self.n_elements):
            nodes = self.element_table[j]
            for i in range(nnodes_per_elem):
                idx_e[j, i] = nodes[i]

        return idx_e

    def _calc_z_using_idx(self, zn, idx):
        if zn.ndim == 1:
            zf = zn[idx].mean(axis=1)
        elif zn.ndim == 2:
            n_steps = zn.shape[0]
            zf = np.zeros((n_steps, idx.shape[0]))
            for step in range(n_steps):
                zf[step, :] = zn[step, idx].mean(axis=1)

        return zf


class _GeometryFMSpectrum(GeometryFM):
    def __init__(
        self,
        node_coordinates,
        element_table,
        codes=None,
        projection=None,
        dfsu_type=None,
        element_ids=None,
        node_ids=None,
        validate=True,
        frequencies=None,
        directions=None,
    ) -> None:
        super().__init__(
            node_coordinates=node_coordinates,
            element_table=element_table,
            codes=codes,
            projection=projection,
            dfsu_type=dfsu_type,
            element_ids=element_ids,
            node_ids=node_ids,
            validate=validate,
        )

        self._frequencies = frequencies
        self._directions = directions

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


class GeometryFMAreaSpectrum(_GeometryFMSpectrum):
    def isel(self, idx=None, axis="elements"):
        return self.elements_to_geometry(elements=idx)

    def elements_to_geometry(
        self, elements
    ) -> Union["GeometryFMAreaSpectrum", GeometryFMPointSpectrum]:
        """export a selection of elements to new flexible file geometry
        Parameters
        ----------
        elements : list(int)
            list of element ids
        Returns
        -------
        GeometryFMAreaSpectrum or GeometryFMPointSpectrum
            which can be used for further extraction or saved to file
        """
        elements = np.atleast_1d(elements)
        if len(elements) == 1:
            coords = self.element_coordinates[elements[0], :]
            return GeometryFMPointSpectrum(
                frequencies=self._frequencies,
                directions=self._directions,
                x=coords[0],
                y=coords[1],
            )

        elements = np.sort(elements)  # make sure elements are sorted!
        node_ids, elem_tbl = self._get_nodes_and_table_for_elements(elements)
        node_coords = self.node_coordinates[node_ids]
        codes = self.codes[node_ids]

        geom = GeometryFMAreaSpectrum(
            node_coordinates=node_coords,
            codes=codes,
            node_ids=node_ids,
            projection=self.projection_string,
            element_table=elem_tbl,
            element_ids=self.element_ids[elements],
            frequencies=self._frequencies,
            directions=self._directions,
        )
        geom._reindex()
        geom._type = self._type
        return geom


class GeometryFMLineSpectrum(_GeometryFMSpectrum):
    def isel(self, idx=None, axis="node"):
        return self._nodes_to_geometry(nodes=idx)

    def _nodes_to_geometry(self, nodes) -> "GeometryFM":
        """export a selection of nodes to new flexible file geometry
        Note: takes only the elements for which all nodes are selected
        Parameters
        ----------
        nodes : list(int)
            list of node ids
        Returns
        -------
        UnstructuredGeometry
            which can be used for further extraction or saved to file
        """
        nodes = np.atleast_1d(nodes)
        if len(nodes) == 1:
            coords = self.node_coordinates[nodes[0], :2]
            return GeometryFMPointSpectrum(
                frequencies=self._frequencies,
                directions=self._directions,
                x=coords[0],
                y=coords[1],
            )

        elements = []
        for j, el_nodes in enumerate(self.element_table):
            if np.all(np.isin(el_nodes, nodes)):
                elements.append(j)

        assert len(elements) > 0, "no elements found"
        elements = np.sort(elements)  # make sure elements are sorted!

        node_ids, elem_tbl = self._get_nodes_and_table_for_elements(elements)
        node_coords = self.node_coordinates[node_ids]
        codes = self.codes[node_ids]

        geom = GeometryFMLineSpectrum(
            node_coordinates=node_coords,
            codes=codes,
            node_ids=node_ids,
            projection=self.projection_string,
            element_table=elem_tbl,
            element_ids=self.element_ids[elements],
            frequencies=self._frequencies,
            directions=self._directions,
        )
        geom._reindex()
        geom._type = self._type
        return geom
