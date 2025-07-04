from __future__ import annotations
from typing import Any, Sequence


import numpy as np
from mikecore.DfsuFile import DfsuFileType


from ._geometry import _Geometry

from ._FM_geometry import _GeometryFM, GeometryFM2D


class GeometryFMPointSpectrum(_Geometry):
    """Flexible mesh point spectrum."""

    def __init__(
        self,
        frequencies: np.ndarray | None = None,
        directions: np.ndarray | None = None,
        x: float | None = None,
        y: float | None = None,
    ) -> None:
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
    def default_dims(self) -> tuple[str, ...]:
        if self.directions is None:
            return ("frequency",)
        else:
            return ("direction", "frequency")

    @property
    def is_layered(self) -> bool:
        return False

    def __repr__(self) -> str:
        txt = f"Point Spectrum Geometry(frequency:{self.n_frequencies}, direction:{self.n_directions}"
        if self.x is not None:
            txt = txt + f", x:{self.x:.5f}, y:{self.y:.5f}"
        return txt + ")"

    @property
    def n_frequencies(self) -> int:
        """Number of frequencies."""
        return 0 if self.frequencies is None else len(self.frequencies)

    @property
    def frequencies(self) -> np.ndarray | None:
        """Frequency axis."""
        return self._frequencies

    @property
    def n_directions(self) -> int:
        """Number of directions."""
        return 0 if self.directions is None else len(self.directions)

    @property
    def directions(self) -> np.ndarray | None:
        """Directional axis."""
        return self._directions


class _GeometryFMSpectrum(_GeometryFM):
    def __init__(
        self,
        node_coordinates: np.ndarray,
        element_table: Any,
        codes: np.ndarray | None = None,
        projection: str = "LONG/LAT",
        dfsu_type: DfsuFileType | None = None,
        element_ids: np.ndarray | None = None,
        node_ids: np.ndarray | None = None,
        validate: bool = True,
        frequencies: np.ndarray | None = None,
        directions: np.ndarray | None = None,
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

        self._frequencies = frequencies
        self._directions = directions

    @property
    def n_frequencies(self) -> int:
        """Number of frequencies."""
        return 0 if self.frequencies is None else len(self.frequencies)

    @property
    def is_layered(self) -> bool:
        return False

    @property
    def frequencies(self) -> np.ndarray | None:
        """Frequency axis."""
        return self._frequencies

    @property
    def n_directions(self) -> int:
        """Number of directions."""
        return 0 if self.directions is None else len(self.directions)

    @property
    def directions(self) -> np.ndarray | None:
        """Directional axis."""
        return self._directions

    @property
    def is_spectral(self) -> bool:
        """Type is spectral dfsu (point, line or area spectrum)."""
        return True


class GeometryFMAreaSpectrum(_GeometryFMSpectrum, GeometryFM2D):
    """Flexible mesh area spectrum geometry."""

    def isel(  # type: ignore
        self, idx: Sequence[int], **kwargs: Any
    ) -> GeometryFMPointSpectrum | GeometryFMAreaSpectrum:
        return self.elements_to_geometry(elements=idx)

    def elements_to_geometry(  # type: ignore
        self, elements: Sequence[int], keepdims: bool = False
    ) -> GeometryFMPointSpectrum | GeometryFMAreaSpectrum:
        """export a selection of elements to new flexible file geometry
        Parameters.
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
        elements = np.atleast_1d(elements)  # type: ignore
        if len(elements) == 1:
            coords = self.element_coordinates[elements[0], :]
            return GeometryFMPointSpectrum(
                frequencies=self._frequencies,
                directions=self._directions,
                x=coords[0],
                y=coords[1],
            )

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


class GeometryFMLineSpectrum(_GeometryFMSpectrum):
    """Flexible mesh line spectrum geometry."""

    @staticmethod
    def create_dummy_coordinates(
        n_nodes: int, frequencies: np.ndarray, directions: np.ndarray
    ) -> GeometryFMLineSpectrum:
        # boogus x, y, z 1..n, 1..n, -10
        node_coordinates = np.zeros((n_nodes, 3), dtype=float)
        node_coordinates[:, 0] = np.arange(1, n_nodes + 1)  # x
        node_coordinates[:, 1] = np.arange(1, n_nodes + 1)  # y
        node_coordinates[:, 2] = -10.0  # z

        # element table with nodes [0,1], [1,2], ..., [n-2,n-1]
        element_table = np.zeros((n_nodes - 1, 2), dtype=int)
        for i in range(n_nodes - 1):
            element_table[i, 0] = i
            element_table[i, 1] = i + 1
        return GeometryFMLineSpectrum(
            node_coordinates=node_coordinates,
            element_table=element_table,
            codes=np.zeros(n_nodes, dtype=int),
            dfsu_type=DfsuFileType.DfsuSpectral1D,
            projection="LONG/LAT",
            frequencies=frequencies,
            directions=directions,
        )

    def isel(  # type: ignore
        self, idx: Sequence[int], axis: str = "node"
    ) -> GeometryFMPointSpectrum | GeometryFMLineSpectrum:
        return self._nodes_to_geometry(nodes=idx)

    def _get_nodes_and_table_for_elements(
        self, elements: np.ndarray | list[int]
    ) -> tuple[Any, Any]:
        elem_tbl = np.empty(len(elements), dtype=np.dtype("O"))

        for j, eid in enumerate(elements):
            elem_tbl[j] = np.asarray(self.element_table[eid])

        nodes = np.unique(np.hstack(elem_tbl))  # type: ignore
        return nodes, elem_tbl

    def _nodes_to_geometry(  # type: ignore
        self, nodes: Sequence[int]
    ) -> GeometryFMPointSpectrum | GeometryFMLineSpectrum:
        """export a selection of nodes to new flexible file geometry
        Note: takes only the elements for which all nodes are selected
        Parameters.
        ----------
        nodes : list(int)
            list of node ids
        Returns
        -------
        GeometryFMPointSpectrum | GeometryFMLineSpectrum
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

        node_ids, elem_tbl = self._get_nodes_and_table_for_elements(elements)
        node_coords = self.node_coordinates[node_ids]
        codes = self.codes[node_ids]

        geom = GeometryFMLineSpectrum(
            node_coordinates=node_coords,
            codes=codes,
            node_ids=node_ids,
            projection=self.projection_string,
            dfsu_type=self._type,
            element_table=elem_tbl,
            element_ids=self.element_ids[elements],
            frequencies=self._frequencies,
            directions=self._directions,
            reindex=True,
        )
        return geom
