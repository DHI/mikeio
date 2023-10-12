from __future__ import annotations
import warnings
from collections import namedtuple
from functools import cached_property
from typing import Collection, Optional, List

import numpy as np
from mikecore.DfsuFile import DfsuFileType  # type: ignore
from mikecore.eum import eumQuantity  # type: ignore
from mikecore.MeshBuilder import MeshBuilder  # type: ignore
from scipy.spatial import cKDTree

from ..eum import EUMType, EUMUnit
from ..exceptions import OutsideModelDomainError
from .._interpolation import get_idw_interpolant, interp2d
from ._FM_utils import (
    _get_node_centered_data,
    _plot_map,
    BoundaryPolylines,
    _set_xy_label_by_projection,  # TODO remove
    _to_polygons,  # TODO remove
)
from ._geometry import GeometryPoint2D, _Geometry

from ._grid_geometry import Grid2D
from ._utils import xy_to_bbox


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

    def __init__(self, geometry) -> None:
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
        import matplotlib.pyplot as plt  # type: ignore

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        return ax

    def _plot_FM_map(self, ax, **kwargs):

        if "title" not in kwargs:
            kwargs["title"] = "Bathymetry"

        plot_type = kwargs.pop("plot_type")

        g = self.g

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

        from matplotlib.collections import PatchCollection  # type: ignore

        ax = self._get_ax(ax=ax, figsize=figsize)
        ax.set_aspect(self._plot_aspect())

        patches = _to_polygons(self.g.node_coordinates, self.g.element_table)
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


class _GeometryFM(_Geometry):
    def __init__(
        self,
        node_coordinates,
        element_table,
        codes=None,
        projection=None,
        dfsu_type=None,  # TODO should this be mandatory?
        element_ids=None,
        node_ids=None,
        validate=True,
        reindex=False,
    ) -> None:
        super().__init__(projection=projection)
        self.node_coordinates = np.asarray(node_coordinates)

        n_nodes = len(node_coordinates)
        self._codes = (
            np.zeros((n_nodes,), dtype=int) if codes is None else np.asarray(codes)
        )

        self._node_ids = (
            np.arange(len(self._codes)) if node_ids is None else np.asarray(node_ids)
        )

        self._type = dfsu_type

        self.element_table, self._element_ids = self._check_elements(
            element_table=element_table,
            element_ids=element_ids,
            validate=validate,
        )

        if reindex:
            self._reindex()

    def _check_elements(self, element_table, element_ids=None, validate=True):

        if validate:
            max_node_id = self._node_ids.max()
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

        if element_ids is None:
            element_ids = np.arange(len(element_table))
        element_ids = np.asarray(element_ids)

        return element_table, element_ids

    def _reindex(self):
        new_node_ids = np.arange(self.n_nodes)
        new_element_ids = np.arange(self.n_elements)
        node_dict = dict(zip(self._node_ids, new_node_ids))
        for eid in range(self.n_elements):
            elem_nodes = self.element_table[eid]
            new_elem_nodes = np.zeros_like(elem_nodes)
            for jn, idx in enumerate(elem_nodes):
                new_elem_nodes[jn] = node_dict[idx]
            self.element_table[eid] = new_elem_nodes

        self._node_ids = new_node_ids
        self._element_ids = new_element_ids

    @property
    def n_nodes(self) -> int:
        """Number of nodes"""
        return len(self._node_ids)

    @property
    def node_ids(self):
        return self._node_ids

    @property
    def n_elements(self) -> int:
        """Number of elements"""
        return len(self._element_ids)

    @property
    def element_ids(self):
        return self._element_ids

    @property
    def _nc(self):
        return self.node_coordinates

    @cached_property
    def max_nodes_per_element(self):
        """The maximum number of nodes for an element"""
        maxnodes = 0
        for local_nodes in self.element_table:
            n = len(local_nodes)
            if n > maxnodes:
                maxnodes = n
        return maxnodes

    @property
    def codes(self):
        """Node codes of all nodes (0=water, 1=land, 2...=open boundaries)"""
        return self._codes

    @codes.setter
    def codes(self, v):
        if len(v) != self.n_nodes:
            raise ValueError(f"codes must have length of nodes ({self.n_nodes})")
        self._codes = np.array(v, dtype=np.int32)


class GeometryFM2D(_GeometryFM):
    def __init__(
        self,
        node_coordinates,
        element_table,
        codes=None,
        projection=None,
        dfsu_type=DfsuFileType.Dfsu2D,  # Reasonable default?
        element_ids=None,
        node_ids=None,
        validate=True,
        reindex=False,
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
            reindex=reindex,
        )

        self.plot = _GeometryFMPlotter(self)

    def __str__(self) -> str:

        return f"{self.type_name} ({self.n_elements} elements, {self.n_nodes} nodes)"

    def __repr__(self):
        return (
            f"Flexible Mesh Geometry: {self._type.name}\n"
            f"number of nodes: {self.n_nodes}\n"
            f"number of elements: {self.n_elements}\n"
            f"projection: {self.projection_string}"
        )

    @staticmethod
    def _point_in_polygon(xn: np.ndarray, yn: np.ndarray, xp: float, yp: float) -> bool:
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

    @property
    def type_name(self):
        """Type name, e.g. Mesh, Dfsu2D"""
        return self._type.name if self._type else "Mesh"

    @property
    def ndim(self) -> int:
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
        return False

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

    @cached_property
    def element_coordinates(self):
        """Center coordinates of each element"""
        return self._calc_element_coordinates()

    @cached_property
    def _tree2d(self):
        xy = self.element_coordinates[:, :2]
        return cKDTree(xy)

    def _calc_element_coordinates(self):
        element_table = self.element_table

        n_elements = len(element_table)
        ec = np.empty([n_elements, 3])

        # pre-allocate for speed
        maxnodes = 4
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

            x, y, z = self.node_coordinates[idx[:nnodes]].T

            xcoords[:nnodes, j] = x
            ycoords[:nnodes, j] = y
            zcoords[:nnodes, j] = z

        ec[:, 0] = np.sum(xcoords, axis=0) / nnodes_per_elem
        ec[:, 1] = np.sum(ycoords, axis=0) / nnodes_per_elem
        ec[:, 2] = np.sum(zcoords, axis=0) / nnodes_per_elem

        return ec

    def find_nearest_elements(self, x, y=None, n_nearest=1, return_distances=False):
        """Find index of nearest elements (optionally for a list)

        Parameters
        ----------
        x: float or array(float)
            X coordinate(s) (easting or longitude)
        y: float or array(float)
            Y coordinate(s) (northing or latitude)
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


        See Also
        --------
        find_index : find element indicies for points or an area
        """
        idx, d2d = self._find_n_nearest_2d_elements(x, y, n=n_nearest)

        if return_distances:
            return idx, d2d

        return idx

    def get_2d_interpolant(
        self,
        xy,
        n_nearest: int = 5,
        extrapolate: bool = False,
        p: int = 2,
        radius: Optional[float] = None,
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
                weights[~self.contains(xy)] = np.nan # type: ignore
        elif n_nearest > 1:
            weights = get_idw_interpolant(dists, p=p)
            if not extrapolate:
                weights[~self.contains(xy), :] = np.nan # type: ignore
        else:
            ValueError("n_nearest must be at least 1")

        if radius is not None:
            weights[dists > radius] = np.nan # type: ignore

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

    def _find_n_nearest_2d_elements(self, x, y=None, n=1):

        # TODO

        if n > self.n_elements:
            raise ValueError(
                f"Cannot find {n} nearest! Number of elements: {self.n_elements}"
            )

        if y is None:
            p = x
            if (not np.isscalar(x)) and (np.ndim(x) == 2):
                p = x[:, 0:2]
        else:
            p = np.array((x, y)).T
        d, elem_id = self._tree2d.query(p, k=n)
        return elem_id, d

    def _find_element_2d(self, coords: np.ndarray):

        points_outside = []

        coords = np.atleast_2d(coords)
        nc = self.node_coordinates

        few_nearest, _ = self._find_n_nearest_2d_elements(
            coords, n=min(self.n_elements, 2)
        )
        ids = np.atleast_2d(few_nearest)[:, 0]  # first guess

        for k in range(len(ids)):
            # step 1: is nearest element = element containing point?
            nodes = self.element_table[ids[k]]
            element_found = self._point_in_polygon(
                nc[nodes, 0], nc[nodes, 1], coords[k, 0], coords[k, 1]
            )

            # step 2: if not, then try second nearest point
            if not element_found and self.n_elements > 1:
                candidate = few_nearest[k, 1]
                assert np.isscalar(candidate)
                nodes = self.element_table[candidate]
                element_found = self._point_in_polygon(
                    nc[nodes, 0], nc[nodes, 1], coords[k, 0], coords[k, 1]
                )
                ids[k] = few_nearest[k, 1]

            # step 3: if not, then try with *many* more points
            if not element_found and self.n_elements > 1:
                many_nearest, _ = self._find_n_nearest_2d_elements(
                    coords[k, :],
                    n=min(self.n_elements, 10),  # TODO is 10 enough?
                )
                for p in many_nearest[2:]:  # we have already tried the two first above
                    nodes = self.element_table[p]
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

        nc = self.node_coordinates

        few_nearest, _ = self._find_n_nearest_2d_elements(
            x=x, y=y, n=min(self.n_elements, 10)
        )

        for idx in few_nearest:
            nodes = self.element_table[idx]
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
        nc = self.node_coordinates
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

    @cached_property
    def boundary_polylines(self) -> BoundaryPolylines:
        """Lists of closed polylines defining domain outline"""
        return self._get_boundary_polylines()

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
        import matplotlib.path as mp  # type: ignore

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
        return self.contains(pt)[0]

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
            xy = self.node_coordinates[polyline, :2]
            area = (
                np.dot(xy[:, 1], np.roll(xy[:, 0], 1))
                - np.dot(xy[:, 0], np.roll(xy[:, 1], 1))
            ) * 0.5
            poly_line = np.asarray(polyline)
            xy = self.node_coordinates[poly_line, 0:2]
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
        element_table = self.element_table

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
    ) -> "GeometryFM2D" | GeometryPoint2D:
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
            return self.elements_to_geometry(elements=idx, keepdims=keepdims)

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

    @staticmethod
    def _inside_polygon(polygon, xy):
        import matplotlib.path as mp

        if polygon.ndim == 1:
            polygon = np.column_stack((polygon[0::2], polygon[1::2]))
        return mp.Path(polygon).contains_points(xy)

    def _elements_in_area(self, area):
        """Find 2d element ids of elements inside area"""
        if self._area_is_bbox(area):
            x0, y0, x1, y1 = area
            xc = self.element_coordinates[:, 0]
            yc = self.element_coordinates[:, 1]
            mask = (xc >= x0) & (xc <= x1) & (yc >= y0) & (yc <= y1)
        elif self._area_is_polygon(area):
            polygon = np.array(area)
            xy = self.element_coordinates[:, :2]
            mask = self._inside_polygon(polygon, xy)
        else:
            raise ValueError("'area' must be bbox [x0,y0,x1,y1] or polygon")

        return np.where(mask)[0]

    def _nodes_to_geometry(self, nodes) -> "GeometryFM2D" | GeometryPoint2D:
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

        return GeometryFM2D(
            node_coordinates=node_coords,
            codes=codes,
            node_ids=node_ids,
            dfsu_type=self._type,
            projection=self.projection_string,
            element_table=elem_tbl,
            element_ids=self.element_ids[elements],
            reindex=True,
        )

    def elements_to_geometry(
        self, elements: int | Collection[int], keepdims=False
    ) -> "GeometryFM2D" | GeometryPoint2D:
        
        if isinstance(elements, (int,np.integer)):
            sel_elements : List[int] = [elements]
        else:
            sel_elements = list(elements)
        if len(sel_elements) == 1 and not keepdims:
            x, y, _ = self.element_coordinates[sel_elements.pop(), :]

            return GeometryPoint2D(x=x, y=y, projection=self.projection)

        sorted_elements = np.sort(
            sel_elements
        )  # make sure elements are sorted! # TODO is this necessary? If so, should be done in the initialiser

        # extract information for selected elements

        node_ids, elem_tbl = self._get_nodes_and_table_for_elements(sorted_elements)
        node_coords = self.node_coordinates[node_ids]
        codes = self.codes[node_ids]
        elem_ids = self.element_ids[sorted_elements]

        return GeometryFM2D(
            node_coordinates=node_coords,
            codes=codes,
            node_ids=node_ids,
            projection=self.projection_string,
            element_table=elem_tbl,
            element_ids=elem_ids,
            dfsu_type=self._type,
            reindex=True,
        )

    def _get_nodes_and_table_for_elements(self, elements):
        """list of nodes and element table for a list of elements

        Parameters
        ----------
        elements : np.array(int)
            array of element ids

        Returns
        -------
        np.array(int)
            array of node ids (unique)
        list(list(int))
            element table with a list of nodes for each element
        """
        elem_tbl = np.empty(len(elements), dtype=np.dtype("O"))

        for j, eid in enumerate(elements):
            elem_tbl[j] = np.asarray(self.element_table[eid])

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
        geometry = self
        nc = geometry.node_coordinates
        ec = geometry.element_coordinates
        elem_table = geometry.element_table
        return _get_node_centered_data(nc, elem_table, ec, data, extrapolate)

    def to_shapely(self):
        """Export mesh as shapely MultiPolygon

        Returns
        -------
        shapely.geometry.MultiPolygon
            polygons with mesh elements
        """
        from shapely.geometry import MultiPolygon, Polygon  # type: ignore

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

        nc = self.node_coordinates
        builder.SetNodes(nc[:, 0], nc[:, 1], nc[:, 2], self.codes)
        # builder.SetNodeIds(geom2d.node_ids+1)
        # builder.SetElementIds(geom2d.elements+1)
        element_table_MZ = [np.asarray(row) + 1 for row in self.element_table]
        builder.SetElements(element_table_MZ)
        builder.SetProjection(self.projection_string)
        quantity = eumQuantity.Create(EUMType.Bathymetry, EUMUnit.meter)
        builder.SetEumQuantity(quantity)
        newMesh = builder.CreateMesh()
        newMesh.Write(outfilename)


class GeometryFM(GeometryFM2D):
    """Deprecated, use GeometryFM2D instead"""

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
        reindex=False,
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
            reindex=reindex,
        )

        warnings.warn("GeometryFM is deprecated, use GeometryFM2D instead")


class _GeometryFMSpectrum(GeometryFM2D):
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
        reindex=False,
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
            reindex=reindex,
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


# TODO reconsider inheritance to avoid overriding method signature
class GeometryFMAreaSpectrum(_GeometryFMSpectrum):
    def isel(self, idx=None, axis="elements"):
        return self.elements_to_geometry(elements=idx)

    def elements_to_geometry(
        self, elements, keepdims=False
    ):
        """export a selection of elements to new flexible file geometry
        Parameters
        ----------
        elements : list(int)
            list of element ids
        keepdims: bool
            Not used
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
            reindex=True,
        )
        geom._type = self._type
        return geom


# TODO this inherits indirectly from GeometryFM2D, which is not ideal
class GeometryFMLineSpectrum(_GeometryFMSpectrum):
    def isel(self, idx=None, axis="node"):
        return self._nodes_to_geometry(nodes=idx)

    def _nodes_to_geometry(
        self, nodes
    ):
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
            reindex=True,
        )
        geom._type = self._type  # TODO
        return geom
