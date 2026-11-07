from __future__ import annotations
from functools import cached_property
from pathlib import Path

from typing import TYPE_CHECKING, Any, Iterable, Literal, Sequence
import warnings
from numbers import Integral

from matplotlib.axes import Axes
import numpy as np
from mikecore.DfsuFile import DfsuFileType


from ._FM_geometry import Area, GeometryFM2D, _GeometryFM, _points_in_area
from ._geometry import GeometryPoint3D

from ._FM_plot import _plot_vertical_profile, BoundaryPolygons

from ._distance import relative_cumulative_distance

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

Layer = Literal["all", "bottom", "top"]


class _Layering:
    """Column and layer structure of a layered mesh.

    The elements of a column have consecutive indices, bottom-up. A column is
    identified by its index in `top_elements`.
    """

    def __init__(
        self,
        *,
        element_table: np.ndarray,
        element_ids: np.ndarray,
        node_coordinates: np.ndarray,
        n_layers: int,
        n_sigma: int,
    ) -> None:
        self._element_table = element_table
        self._element_ids = element_ids
        self._node_coordinates = node_coordinates
        self.n_layers = n_layers
        self.n_sigma_layers = n_sigma

    @property
    def n_z_layers(self) -> int:
        return self.n_layers - self.n_sigma_layers

    @cached_property
    def top_elements(self) -> np.ndarray:
        if self.n_z_layers == 0:
            n_elements = len(self._element_table)
            return np.arange(self.n_sigma_layers - 1, n_elements, self.n_sigma_layers)
        return _find_top_layer_elements(self._element_table)

    @cached_property
    def n_layers_per_column(self) -> np.ndarray:
        top_elems = self.top_elements
        below = np.empty_like(top_elems)
        below[0] = -1
        below[1:] = top_elems[:-1]
        return top_elems - below

    @cached_property
    def bottom_elements(self) -> np.ndarray:
        return self.top_elements - self.n_layers_per_column + 1

    @cached_property
    def _association(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        e2_to_e3: list[
            np.ndarray
        ] = []  # for each column: its element ids, bottom to top
        index2d: list[int] = []  # for each element: its column
        layerid: list[int] = []  # for each element: its layer number
        global_layer_ids = np.arange(self.n_layers)
        for j, (bot, top) in enumerate(zip(self.bottom_elements, self.top_elements)):
            col = np.arange(bot, top + 1)
            e2_to_e3.append(col)
            index2d.extend([j] * len(col))
            # columns shorter than n_layers lack the lowest z-layers
            layerid.extend(global_layer_ids[-len(col) :])

        return (
            np.array(e2_to_e3, dtype=object),
            np.array(index2d),
            np.array(layerid),
        )

    @property
    def e2_e3_table(self) -> np.ndarray:
        return self._association[0]

    @property
    def elem2d_ids(self) -> np.ndarray:
        return self._association[1]

    @property
    def layer_ids(self) -> np.ndarray:
        return self._association[2]

    def get_layer_elements(self, layers: int | Layer | Sequence[int]) -> np.ndarray:
        if isinstance(layers, str):
            if layers == "top":
                return self.top_elements
            elif layers == "bottom":
                return self.bottom_elements
            else:
                raise ValueError(
                    f"layers '{layers}' not recognized ('top', 'bottom' or integer)"
                )

        if not np.isscalar(layers):
            if not isinstance(layers, Iterable):
                raise TypeError(_layers_type_error(layers))
            elem_ids = []
            for layer in layers:
                if not _is_integer(layer):
                    raise TypeError(_layers_type_error(layer))
                elem_ids.append(self.get_layer_elements(int(layer)))
            return np.sort(np.concatenate(elem_ids, axis=0))

        n_lay = self.n_layers

        # numpy integers too: n_layers and layer_ids are numpy integers
        if not _is_integer(layers):
            raise TypeError(_layers_type_error(layers))
        layer = int(layers)
        if layer < (-n_lay) or layer >= n_lay:
            raise Exception(
                f"Layer {layer!r} not allowed; must be between -{n_lay} and {n_lay - 1}"
            )

        if layer < 0:
            layer = layer + n_lay

        return self._element_ids[self.layer_ids == layer]

    @cached_property
    def dz(self) -> np.ndarray:
        """Height of each element, from the static node z."""
        zn = self._node_coordinates[:, 2]
        dz = np.full(len(self._element_table), fill_value=np.nan)
        for j, nodes in enumerate(self._element_table):
            halfn = len(nodes) // 2
            dz[j] = np.mean(zn[nodes[halfn:]]) - np.mean(zn[nodes[:halfn]])
        return dz

    def find_elements_at_z(
        self,
        columns: np.ndarray,
        z: np.ndarray | float,
        element_coordinates: np.ndarray,
    ) -> np.ndarray:
        """Element in each of `columns` containing the matching z value."""
        columns = np.atleast_1d(columns)
        z_vec = np.full(columns.shape, fill_value=z) if np.isscalar(z) else z
        elem3d = np.full_like(columns, fill_value=-1)
        for j, column in enumerate(columns):
            col = np.asarray(self.e2_e3_table[column], dtype=int)
            idx = self._z_idx_in_column(col, z_vec[j], element_coordinates)  # type: ignore[index]
            elem3d[j] = col[idx]
        return elem3d

    def _z_idx_in_column(
        self, col: np.ndarray, z: float, element_coordinates: np.ndarray
    ) -> int:
        dz = self.dz[col]
        z_col = element_coordinates[col, 2]
        z_face = np.append(z_col - dz / 2, z_col[-1] + dz[-1] / 2)
        if z < z_face[0] or z > z_face[-1]:
            xy = tuple(element_coordinates[col[0], :2])
            raise ValueError(
                f"z value '{z}' is outside water column [{z_face[0]},{z_face[-1]}] in point x,y={xy}"
            )
        return int(np.searchsorted(z_face, z)) - 1

    def elements_in_columns(self, columns: np.ndarray) -> np.ndarray:
        if len(columns) == 0:
            return np.array([], dtype=int)
        return np.hstack(list(self.e2_e3_table[columns])).astype(int)

    def layers_spanned(self, elements: Sequence[int]) -> tuple[int, int]:
        """Number of layers and of sigma layers in a selection of elements.

        Bottom elements of different columns can sit in different layers, but
        together they form a single layer.
        """
        layer_ids = np.unique(self.layer_ids[elements])
        if len(layer_ids) > 1 and np.all(np.isin(elements, self.bottom_elements)):
            return 1, 1
        n_sigma = int(np.sum(layer_ids >= self.n_z_layers))
        return len(layer_ids), n_sigma

    def is_single_column(self, elements: Sequence[int]) -> bool:
        return len(np.unique(self.elem2d_ids[elements])) == 1


def _find_top_layer_elements(element_table: np.ndarray) -> np.ndarray:
    """Indices of the upper-most element of each column.

    For 3D elements the top half of the nodes of an element are the bottom
    half of the element above it. For 2D vertical elements the shared nodes
    appear in reverse order in the element above.
    """
    top_elements = []
    for i in range(len(element_table) - 1):
        elmt1 = element_table[i]
        elmt2 = element_table[i + 1]

        if len(elmt1) != len(elmt2):
            top_elements.append(i)
            continue

        n_half = len(elmt1) // 2
        for j in range(n_half):
            above = elmt2[j] if n_half > 2 else elmt2[(n_half - 1) - j]
            if elmt1[j + n_half] != above:
                top_elements.append(i)
                break

    top_elements.append(len(element_table) - 1)
    return np.array(top_elements, dtype=np.int32)


def _nodes_and_table(
    element_table: np.ndarray,
    elements: Sequence[int] | np.ndarray,
    node_layers: Layer = "all",
) -> tuple[Any, Any]:
    """Unique node ids and element table for a list of elements.

    node_layers 'bottom' or 'top' keeps only that half of each element's nodes.
    """
    elem_tbl = np.empty(len(elements), dtype=np.dtype("O"))
    for j, eid in enumerate(elements):
        elem_nodes = element_table[eid]
        halfn = len(elem_nodes) // 2
        if node_layers == "bottom":
            elem_nodes = elem_nodes[:halfn]
        elif node_layers == "top":
            elem_nodes = elem_nodes[halfn:]
        elem_tbl[j] = elem_nodes

    nodes = np.unique(np.hstack(list(elem_tbl)))
    return nodes, elem_tbl


def _as_element_list(elements: int | Sequence[int] | np.ndarray) -> list[int]:
    if isinstance(elements, (int, np.integer)):
        return [int(elements)]
    return list(elements)


def _layered_repr(
    geom: GeometryFM3D | GeometryFMVerticalProfile | GeometryFMVerticalColumn,
) -> str:
    return (
        f"Flexible Mesh Geometry: {geom._type.name}\n"
        f"number of nodes: {geom.n_nodes}\n"
        f"number of elements: {geom.n_elements}\n"
        f"number of layers: {geom.n_layers}\n"
        f"number of sigma layers: {geom.n_sigma_layers}\n"
        f"projection: {geom.projection_string}"
    )


def _subset_kwargs(
    geom: GeometryFM3D | GeometryFMVerticalProfile | GeometryFMVerticalColumn,
    elements: list[int],
    node_layers: Layer = "all",
) -> dict[str, Any]:
    """Constructor arguments for a layered geometry of the selected elements."""
    node_ids, elem_tbl = geom._get_nodes_and_table_for_elements(
        elements, node_layers=node_layers
    )
    n_layers, n_sigma = geom._layering.layers_spanned(elements)
    return dict(
        node_coordinates=geom.node_coordinates[node_ids],
        codes=geom.codes[node_ids],
        node_ids=node_ids,
        projection=geom.projection_string,
        element_table=elem_tbl,
        element_ids=geom._element_ids[elements],
        dfsu_type=geom._type,
        reindex=True,
        n_layers=n_layers,
        n_sigma=n_sigma,
    )


def _parse_points(
    x: float | None,
    y: float | None,
    z: float | None,
    coords: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | float | None]:
    """Points as an (n, 2) xy array plus their z values, if any."""
    if coords is not None:
        coords = np.atleast_2d(coords)
        z_out = coords[:, 2] if coords.shape[1] == 3 else None
        return coords[:, :2], z_out
    return np.vstack((x, y)).T, z  # type: ignore[arg-type]


class GeometryFM3D(_GeometryFM):
    """Flexible 3d mesh geometry."""

    def __init__(
        self,
        *,
        node_coordinates: ArrayLike,
        element_table: ArrayLike,
        codes: np.ndarray | None = None,
        projection: str = "LONG/LAT",
        dfsu_type: DfsuFileType = DfsuFileType.Dfsu3DSigma,
        element_ids: np.ndarray | None = None,
        node_ids: np.ndarray | None = None,
        n_layers: int = 1,
        n_sigma: int | None = None,
        validate: bool = True,
        reindex: bool = False,
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
        self._n_layers = n_layers
        self._n_sigma = n_sigma if n_sigma is not None else n_layers

    @cached_property
    def _layering(self) -> _Layering:
        return _Layering(
            element_table=self.element_table,
            element_ids=self._element_ids,
            node_coordinates=self.node_coordinates,
            n_layers=self._n_layers,
            n_sigma=self._n_sigma,
        )

    def __str__(self) -> str:
        return f"{self._type.name} ({self.n_elements} elements, {self.n_nodes} nodes)"

    def __repr__(self) -> str:
        return _layered_repr(self)

    @property
    def is_layered(self) -> bool:
        """Type is layered dfsu (3d, vertical profile or vertical column)."""
        return True

    @property
    def n_layers(self) -> int:
        """Maximum number of layers."""
        return self._layering.n_layers

    @property
    def n_sigma_layers(self) -> int:
        """Number of sigma layers."""
        return self._layering.n_sigma_layers

    @property
    def n_z_layers(self) -> int:
        """Maximum number of z-layers."""
        return self._layering.n_z_layers

    @property
    def top_elements(self) -> np.ndarray:
        """List of 3d element ids of surface layer."""
        return self._layering.top_elements

    @property
    def bottom_elements(self) -> np.ndarray:
        """List of 3d element ids of bottom layer."""
        return self._layering.bottom_elements

    @property
    def n_layers_per_column(self) -> np.ndarray:
        """List of number of layers for each column."""
        return self._layering.n_layers_per_column

    @property
    def layer_ids(self) -> np.ndarray:
        """The layer number (0=bottom, 1, 2, ...) for each 3d element."""
        return self._layering.layer_ids

    @property
    def e2_e3_table(self) -> np.ndarray:
        """The 2d-to-3d element connectivity table for a 3d object."""
        return self._layering.e2_e3_table

    @property
    def elem2d_ids(self) -> np.ndarray:
        """The associated 2d element id for each 3d element."""
        return self._layering.elem2d_ids

    def get_layer_elements(self, layers: int | Layer | Sequence[int]) -> np.ndarray:
        """3d element ids for one (or more) specific layer(s).

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
        return self._layering.get_layer_elements(layers)

    def _get_nodes_and_table_for_elements(
        self, elements: Sequence[int] | np.ndarray, node_layers: Layer = "all"
    ) -> tuple[Any, Any]:
        return _nodes_and_table(self.element_table, elements, node_layers)

    @cached_property
    def geometry2d(self) -> GeometryFM2D:
        """The 2d horizontal mesh."""
        return self.to_2d_geometry()

    def to_2d_geometry(self) -> GeometryFM2D:
        """extract 2d geometry from 3d geometry.

        Returns
        -------
        GeometryFM2D
            2d geometry

        """
        match self._type:
            case DfsuFileType.Dfsu3DSigmaZ:
                elem_ids = self.top_elements
            case DfsuFileType.Dfsu3DSigma:
                elem_ids = self.bottom_elements
            case _:
                raise NotImplementedError(
                    f"Conversion to 2D not implemented for {self._type}"
                )

        node_ids, elem_tbl = self._get_nodes_and_table_for_elements(
            elem_ids, node_layers="bottom"
        )
        geom = GeometryFM2D(
            node_coordinates=self.node_coordinates[node_ids],
            codes=self._codes[node_ids],
            node_ids=node_ids,
            projection=self.projection_string,
            element_table=elem_tbl,
            element_ids=elem_ids,
            validate=False,
            dfsu_type=DfsuFileType.Dfsu2D,
            reindex=True,
        )

        # sigma-z: the 2d node z is the lowest bed z of the nodes in its column
        if self._type == DfsuFileType.Dfsu3DSigmaZ:
            zn = geom.node_coordinates[:, 2].copy()
            for j, elem_nodes in enumerate(geom.element_table):
                elem_nodes3d = self.element_table[self.bottom_elements[j]]
                for jn in range(len(elem_nodes)):
                    znj_3d = self.node_coordinates[elem_nodes3d[jn], 2]
                    zn[elem_nodes[jn]] = min(zn[elem_nodes[jn]], znj_3d)
            geom.node_coordinates[:, 2] = zn

        return geom

    def _elements_in_area(self, area: Area) -> np.ndarray:
        """Find element ids of elements inside area."""
        return self._layering.elements_in_columns(
            self.geometry2d._elements_in_area(area)
        )

    @property
    def plot(self) -> None:
        """Not available for GeometryFM3D, use to_2d_geometry().plot instead."""
        raise AttributeError(
            "GeometryFM3D does not support plotting directly. "
            "Use .to_2d_geometry().plot instead."
        )

    @property
    def boundary_polylines(self) -> BoundaryPolygons:
        """Lists of polygons defining domain outline (deprecated, use boundary_polygons)."""
        warnings.warn(
            "boundary_polylines is renamed to boundary_polygons", FutureWarning
        )
        return self.geometry2d.boundary_polygons

    @property
    def boundary_polygons(self) -> BoundaryPolygons:
        """Lists of polygons defining domain outline."""
        return self.geometry2d.boundary_polygons

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Test if a list of points are contained by the horizontal extent of the mesh."""
        return self.geometry2d.contains(points)

    def to_mesh(self, outfilename: str | Path) -> None:
        """Export the horizontal mesh to a mesh file."""
        return self.geometry2d.to_mesh(outfilename)

    def isel(
        self, idx: Sequence[int] | np.ndarray, keepdims: bool = False, **kwargs: Any
    ) -> GeometryFM3D | GeometryFMVerticalColumn | GeometryFM2D | GeometryPoint3D:
        """Export a selection of elements to a new geometry."""
        return self.elements_to_geometry(elements=idx, keepdims=keepdims)

    def elements_to_geometry(
        self,
        elements: int | Sequence[int] | np.ndarray,
        node_layers: Layer = "all",
        keepdims: bool = False,
    ) -> GeometryFM3D | GeometryFMVerticalColumn | GeometryFM2D | GeometryPoint3D:
        """Export a selection of elements to a new geometry.

        A selection within a single layer becomes a 2d mesh, and a selection
        within a single column becomes a vertical column.
        """
        sel = _as_element_list(elements)
        if len(sel) == 1 and not keepdims:
            x, y, z = self.element_coordinates[sel[0], :]
            return GeometryPoint3D(x=x, y=y, z=z, projection=self.projection)

        n_layers, _ = self._layering.layers_spanned(sel)
        if n_layers == 1:
            return self._layer_to_2d_geometry(sel)

        kwargs = _subset_kwargs(self, sel, node_layers=node_layers)
        if self._layering.is_single_column(sel):
            return GeometryFMVerticalColumn(**kwargs)
        return GeometryFM3D(**kwargs)

    def _layer_to_2d_geometry(self, elements: list[int]) -> GeometryFM2D:
        elem2d = self.elem2d_ids[elements]
        geom2d = self.geometry2d
        node_ids, elem_tbl = geom2d._get_nodes_and_table_for_elements(elem2d)
        return GeometryFM2D(
            node_coordinates=geom2d.node_coordinates[node_ids],
            codes=geom2d.codes[node_ids],
            node_ids=node_ids,
            projection=self.projection_string,
            element_table=elem_tbl,
            element_ids=self._element_ids[elem2d],
            dfsu_type=DfsuFileType.Dfsu2D,
            reindex=True,
        )

    def find_index(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        coords: np.ndarray | None = None,
        area: Area | None = None,
        layers: int | Layer | Sequence[int] | None = None,
    ) -> np.ndarray:
        """Find a *set* of 3d element indices for points, an area and/or layers.

        Typically not called directly, but by Dataset/DataArray's
        sel() method.

        Parameters
        ----------
        x: float, optional
            X coordinate (easting or longitude)
        y: float, optional
            Y coordinate (northing or latitude)
        z: float, optional
            Z coordinate (depth, positive upwards)
        coords : np.array(float,float), optional
            As an alternative to specifying x, y (and z) individually,
            the argument coords can be used instead,
            by default None
        area : bbox, polygon or shapely geometry, optional
            Area to be selected: a bounding box (left, lower, right, upper),
            a list of (x, y) polygon vertices, or a
            Polygon or MultiPolygon with `__geo_interface__`, e.g. from shapely
            or geopandas, holes excluded. Elements are selected when their centre is inside the area or on its boundary.
            By default None
        layers : int or str or list[int], optional
            layer(s) to be selected: subset of "all", "top", "bottom",
            "bottom+1", ..., or a layer number (0=bottom, 1, 2, ...),
            by default None

        Returns
        -------
        np.array
            3d element indices

        """
        idx = self._element_ids if layers is None else self.get_layer_elements(layers)

        if any(v is not None for v in (x, y, z, coords)):
            if area is not None:
                raise ValueError(
                    "Coordinates and area cannot be provided at the same time!"
                )
            xy, z_sel = _parse_points(x, y, z, coords)
            columns = self.geometry2d._find_element_2d(coords=xy)
            assert len(columns) == len(xy)
            if z_sel is None:
                idx_3d = self._layering.elements_in_columns(columns)
            else:
                idx_3d = self._layering.find_elements_at_z(
                    columns, z_sel, self.element_coordinates
                )
            idx = np.intersect1d(idx, idx_3d).astype(int)
        elif area is not None:
            idx = np.intersect1d(idx, self._elements_in_area(area))
        elif layers is None:
            raise ValueError(
                "At least one selection argument (x,y,z,coords,area,layers) needs to be provided!"
            )
        return idx


class GeometryFMVerticalProfile(_GeometryFM):
    """Flexible mesh 2d vertical profile geometry."""

    def __init__(
        self,
        *,
        node_coordinates: ArrayLike,
        element_table: ArrayLike,
        codes: np.ndarray | None = None,
        projection: str = "LONG/LAT",
        dfsu_type: DfsuFileType = DfsuFileType.DfsuVerticalProfileSigmaZ,
        element_ids: np.ndarray | None = None,
        node_ids: np.ndarray | None = None,
        n_layers: int = 1,
        n_sigma: int | None = None,
        validate: bool = True,
        reindex: bool = False,
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
        self._n_layers = n_layers
        self._n_sigma = n_sigma if n_sigma is not None else n_layers

    @cached_property
    def _layering(self) -> _Layering:
        return _Layering(
            element_table=self.element_table,
            element_ids=self._element_ids,
            node_coordinates=self.node_coordinates,
            n_layers=self._n_layers,
            n_sigma=self._n_sigma,
        )

    def __str__(self) -> str:
        return f"{self._type.name} ({self.n_elements} elements, {self.n_nodes} nodes)"

    def __repr__(self) -> str:
        return _layered_repr(self)

    @property
    def is_layered(self) -> bool:
        """Type is layered dfsu (3d, vertical profile or vertical column)."""
        return True

    @property
    def n_layers(self) -> int:
        """Maximum number of layers."""
        return self._layering.n_layers

    @property
    def n_sigma_layers(self) -> int:
        """Number of sigma layers."""
        return self._layering.n_sigma_layers

    @property
    def n_z_layers(self) -> int:
        """Maximum number of z-layers."""
        return self._layering.n_z_layers

    @property
    def top_elements(self) -> np.ndarray:
        """List of element ids of surface layer."""
        return self._layering.top_elements

    @property
    def bottom_elements(self) -> np.ndarray:
        """List of element ids of bottom layer."""
        return self._layering.bottom_elements

    @property
    def n_layers_per_column(self) -> np.ndarray:
        """List of number of layers for each column."""
        return self._layering.n_layers_per_column

    @property
    def layer_ids(self) -> np.ndarray:
        """The layer number (0=bottom, 1, 2, ...) for each element."""
        return self._layering.layer_ids

    @property
    def e2_e3_table(self) -> np.ndarray:
        """The element ids of each column, bottom to top."""
        return self._layering.e2_e3_table

    @property
    def elem2d_ids(self) -> np.ndarray:
        """The column index of each element."""
        return self._layering.elem2d_ids

    def get_layer_elements(self, layers: int | Layer | Sequence[int]) -> np.ndarray:
        """Element ids for one (or more) specific layer(s).

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
        return self._layering.get_layer_elements(layers)

    def _get_nodes_and_table_for_elements(
        self, elements: Sequence[int] | np.ndarray, node_layers: Layer = "all"
    ) -> tuple[Any, Any]:
        return _nodes_and_table(self.element_table, elements, node_layers)

    @cached_property
    def plot(self) -> GeometryFMVerticalProfilePlotter:
        """Plot the profile."""
        return GeometryFMVerticalProfilePlotter(self)

    @cached_property
    def relative_element_distance(self) -> np.ndarray:
        """Distance of each element along the profile from its start."""
        ec = self.element_coordinates
        nc0 = self.node_coordinates[0, :2]
        return relative_cumulative_distance(ec, reference=nc0, is_geo=self.is_geo)

    def get_nearest_relative_distance(self, coords: tuple[float, float]) -> float:
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

    def isel(
        self, idx: Sequence[int] | np.ndarray, keepdims: bool = False, **kwargs: Any
    ) -> GeometryFMVerticalProfile | GeometryFMVerticalColumn | GeometryPoint3D:
        """Export a selection of elements to a new geometry."""
        return self.elements_to_geometry(elements=idx, keepdims=keepdims)

    def elements_to_geometry(
        self,
        elements: int | Sequence[int] | np.ndarray,
        keepdims: bool = False,
    ) -> GeometryFMVerticalProfile | GeometryFMVerticalColumn | GeometryPoint3D:
        """Export a selection of elements to a new geometry.

        A selection within a single column becomes a vertical column.
        """
        sel = _as_element_list(elements)
        if len(sel) == 1 and not keepdims:
            x, y, z = self.element_coordinates[sel[0], :]
            return GeometryPoint3D(x=x, y=y, z=z, projection=self.projection)

        kwargs = _subset_kwargs(self, sel)
        if self._layering.is_single_column(sel):
            return GeometryFMVerticalColumn(**kwargs)
        return GeometryFMVerticalProfile(**kwargs)

    def _elements_in_area(self, area: Area) -> np.ndarray:
        """Element ids of the columns whose position is inside area."""
        xy = self.element_coordinates[self.top_elements, :2]
        return self._layering.elements_in_columns(_points_in_area(xy, area))

    def find_index(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        coords: np.ndarray | None = None,
        area: Area | None = None,
        layers: int | Layer | Sequence[int] | None = None,
    ) -> np.ndarray:
        """Find a *set* of element indices for points, an area and/or layers.

        Typically not called directly, but by Dataset/DataArray's
        sel() method.

        Parameters
        ----------
        x: float, optional
            X coordinate (easting or longitude)
        y: float, optional
            Y coordinate (northing or latitude)
        z: float, optional
            Z coordinate (depth, positive upwards)
        coords : np.array(float,float), optional
            As an alternative to specifying x, y (and z) individually,
            the argument coords can be used instead,
            by default None
        area : bbox, polygon or shapely geometry, optional
            Area to be selected: a bounding box (left, lower, right, upper),
            a list of (x, y) polygon vertices, or a
            Polygon or MultiPolygon with `__geo_interface__`, holes excluded;
            selects the columns whose centre is inside the area or on its
            boundary, by default None
        layers : int or str or list[int], optional
            layer(s) to be selected: subset of "all", "top", "bottom",
            "bottom+1", ..., or a layer number (0=bottom, 1, 2, ...),
            by default None

        Returns
        -------
        np.array
            element indices

        """
        idx = self._element_ids if layers is None else self.get_layer_elements(layers)

        if any(v is not None for v in (x, y, z, coords)):
            if area is not None:
                raise ValueError(
                    "Coordinates and area cannot be provided at the same time!"
                )
            xy, z_sel = _parse_points(x, y, z, coords)
            columns = self._find_nearest_column(xy)
            if z_sel is None:
                idx_3d = self._layering.elements_in_columns(columns)
            else:
                idx_3d = self._layering.find_elements_at_z(
                    columns, z_sel, self.element_coordinates
                )
            idx = np.intersect1d(idx, idx_3d)
        elif area is not None:
            idx = np.intersect1d(idx, self._elements_in_area(area))

        return idx

    def _find_nearest_column(self, coords: np.ndarray) -> np.ndarray:
        ec2d = self.element_coordinates[self.top_elements, :2]
        xe, ye = ec2d[:, 0], ec2d[:, 1]
        coords = np.atleast_2d(coords)
        idx = np.zeros(len(coords), dtype=int)
        for j, pt in enumerate(coords):
            x, y = pt[0:2]
            idx[j] = np.argmin((xe - x) ** 2 + (ye - y) ** 2)
        return idx


class GeometryFMVerticalColumn(_GeometryFM):
    """A layered geometry consisting of a single vertical column."""

    def __init__(
        self,
        *,
        node_coordinates: ArrayLike,
        element_table: ArrayLike,
        codes: np.ndarray | None = None,
        projection: str = "LONG/LAT",
        dfsu_type: DfsuFileType = DfsuFileType.Dfsu3DSigma,
        element_ids: np.ndarray | None = None,
        node_ids: np.ndarray | None = None,
        n_layers: int = 1,
        n_sigma: int | None = None,
        validate: bool = True,
        reindex: bool = False,
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
        self._n_layers = n_layers
        self._n_sigma = n_sigma if n_sigma is not None else n_layers

    @cached_property
    def _layering(self) -> _Layering:
        return _Layering(
            element_table=self.element_table,
            element_ids=self._element_ids,
            node_coordinates=self.node_coordinates,
            n_layers=self._n_layers,
            n_sigma=self._n_sigma,
        )

    def __str__(self) -> str:
        return f"{self._type.name} ({self.n_elements} elements, {self.n_nodes} nodes)"

    def __repr__(self) -> str:
        return _layered_repr(self)

    @property
    def is_layered(self) -> bool:
        """Type is layered dfsu (3d, vertical profile or vertical column)."""
        return True

    @property
    def n_layers(self) -> int:
        """Number of layers."""
        return self._layering.n_layers

    @property
    def n_sigma_layers(self) -> int:
        """Number of sigma layers."""
        return self._layering.n_sigma_layers

    @property
    def n_z_layers(self) -> int:
        """Number of z-layers."""
        return self._layering.n_z_layers

    @property
    def top_elements(self) -> np.ndarray:
        """Element id of the surface element, as a list of one."""
        return self._layering.top_elements

    @property
    def bottom_elements(self) -> np.ndarray:
        """Element id of the bottom element, as a list of one."""
        return self._layering.bottom_elements

    @property
    def n_layers_per_column(self) -> np.ndarray:
        """Number of layers, as a list of one."""
        return self._layering.n_layers_per_column

    @property
    def layer_ids(self) -> np.ndarray:
        """The layer number (0=bottom, 1, 2, ...) for each element."""
        return self._layering.layer_ids

    @property
    def e2_e3_table(self) -> np.ndarray:
        """The element ids of the column, bottom to top, as a table of one."""
        return self._layering.e2_e3_table

    @property
    def elem2d_ids(self) -> np.ndarray:
        """The column index of each element (all 0)."""
        return self._layering.elem2d_ids

    def get_layer_elements(self, layers: int | Layer | Sequence[int]) -> np.ndarray:
        """Element ids for one (or more) specific layer(s).

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
        return self._layering.get_layer_elements(layers)

    def _get_nodes_and_table_for_elements(
        self, elements: Sequence[int] | np.ndarray, node_layers: Layer = "all"
    ) -> tuple[Any, Any]:
        return _nodes_and_table(self.element_table, elements, node_layers)

    def isel(
        self, idx: Sequence[int] | np.ndarray, keepdims: bool = False, **kwargs: Any
    ) -> GeometryFMVerticalColumn | GeometryPoint3D:
        """Export a selection of elements to a new geometry."""
        return self.elements_to_geometry(elements=idx, keepdims=keepdims)

    def elements_to_geometry(
        self,
        elements: int | Sequence[int] | np.ndarray,
        keepdims: bool = False,
    ) -> GeometryFMVerticalColumn | GeometryPoint3D:
        """Export a selection of elements to a new geometry."""
        sel = _as_element_list(elements)
        if len(sel) == 1 and not keepdims:
            x, y, z = self.element_coordinates[sel[0], :]
            return GeometryPoint3D(x=x, y=y, z=z, projection=self.projection)
        return GeometryFMVerticalColumn(**_subset_kwargs(self, sel))

    def find_index(
        self,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        coords: np.ndarray | None = None,
        area: Area | None = None,
        layers: int | Layer | Sequence[int] | None = None,
    ) -> np.ndarray:
        """Find a *set* of element indices for z values and/or layers.

        A column has a single horizontal position, so x, y, area and the x,y
        part of coords are rejected.

        Parameters
        ----------
        x: float, optional
            Not supported
        y: float, optional
            Not supported
        z: float, optional
            Z coordinate (depth, positive upwards)
        coords : np.array(float,float,float), optional
            Not supported
        area : (float, float, float, float), optional
            Not supported
        layers : int or str or list[int], optional
            layer(s) to be selected: subset of "all", "top", "bottom",
            "bottom+1", ..., or a layer number (0=bottom, 1, 2, ...),
            by default None

        Returns
        -------
        np.array
            element indices

        """
        if any(v is not None for v in (x, y, coords, area)):
            raise ValueError(
                "A vertical column has a single horizontal position; select with z or layers"
            )
        idx = self._element_ids if layers is None else self.get_layer_elements(layers)
        if z is not None:
            idx_z = self._layering.find_elements_at_z(
                np.zeros(1, dtype=int), z, self.element_coordinates
            )
            idx = np.intersect1d(idx, idx_z)
        return idx

    def calc_ze(self, zn: np.ndarray | None = None) -> np.ndarray:
        """Z-coordinates at element centers."""
        return self._calc_z(zn, self._idx_e)

    def calc_zf(self, zn: np.ndarray | None = None) -> np.ndarray:
        """Z-coordinates at layer faces (horizontal interfaces between layers)."""
        return self._calc_z(zn, self._idx_f)

    def _calc_zee(self, zn: np.ndarray | None = None) -> np.ndarray:
        """Element center z-coords extended with bottom and top face for extrapolation."""
        ze = self.calc_ze(zn)
        zf = self.calc_zf(zn)
        if ze.ndim == 1:
            return np.append(np.insert(ze, 0, zf[0]), zf[-1])
        return np.hstack((zf[:, 0].reshape((-1, 1)), ze, zf[:, -1].reshape((-1, 1))))

    @staticmethod
    def _linear_interp(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
        """Linear interpolation with extrapolation.

        Uses make_interp_spline(k=1) which extrapolates beyond boundaries by default.
        """
        from scipy.interpolate import make_interp_spline  # type: ignore

        return make_interp_spline(x, y, k=1)(x_new)  # type: ignore

    def _interp_values(
        self, zn: np.ndarray, data: np.ndarray, z: np.ndarray
    ) -> np.ndarray:
        """Interpolate to other z values, allow linear extrapolation."""
        ze = self.calc_ze(zn)
        if zn.ndim == 1:
            return self._linear_interp(ze, data, z)
        dati = np.zeros_like(z)
        for j in range(zn.shape[0]):
            dati[j, :] = self._linear_interp(ze[j, :], data[j, :], z[j, :])
        return dati

    @cached_property
    def _idx_e(self) -> np.ndarray:
        """Node indices per element as a 2D array."""
        return np.stack(self.element_table)

    @cached_property
    def _idx_f(self) -> np.ndarray:
        """Node indices per layer face (n_elements + 1 faces)."""
        nnodes_half = len(self.element_table[0]) // 2
        idx_e = self._idx_e
        idx_f = np.zeros((self.n_elements + 1, nnodes_half), dtype=int)
        # Bottom nodes of each element define the lower faces
        idx_f[: self.n_elements, :] = idx_e[:, :nnodes_half]
        # Top nodes of the last element define the top face
        idx_f[self.n_elements, :] = idx_e[-1, nnodes_half:]
        return idx_f

    def _calc_z(self, zn: np.ndarray | None, idx: np.ndarray) -> np.ndarray:
        if zn is None:
            zn = self.node_coordinates[:, 2]
        return zn[..., idx].mean(axis=-1)


GeometryFMLayered = GeometryFM3D | GeometryFMVerticalProfile | GeometryFMVerticalColumn
"""Geometries made of layered columns."""


class GeometryFMVerticalProfilePlotter:
    """Plot a vertical profile geometry, e.g. geometry.plot()."""

    def __init__(self, geometry: GeometryFMVerticalProfile) -> None:
        self.g = geometry

    def __call__(
        self,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> Axes:
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        x = self.g.node_coordinates[:, 0]
        y = self.g.node_coordinates[:, 1]
        ax.plot(x, y, **kwargs)
        return ax

    def mesh(self, title: str = "Mesh", edge_color: str = "0.5", **kwargs: Any) -> Axes:
        """Plot the mesh of the vertical profile."""
        v = np.full_like(self.g.element_coordinates[:, 0], np.nan)
        return _plot_vertical_profile(
            node_coordinates=self.g.node_coordinates,
            element_table=self.g.element_table,
            values=v,
            is_geo=self.g.is_geo,
            title=title,
            add_colorbar=False,
            edge_color=edge_color,
            vmin=0.0,
            vmax=1.0,
            **kwargs,
        )


def _is_integer(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _layers_type_error(value: object) -> str:
    return (
        f"layers must be an integer, 'top', 'bottom' or a list of integers, "
        f"not {type(value).__name__} {value!r}"
    )
