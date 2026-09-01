from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING


import numpy as np


from mikecore.MeshFile import MeshFile


from ..spatial import GeometryFM2D
from ._topology import (
    get_elements_from_source,
    get_nodes_from_source,
)

if TYPE_CHECKING:
    from shapely.geometry import MultiPolygon


class Mesh:
    """The Mesh class is initialized with a mesh file.

    Parameters
    ---------
    filename: str
        mesh filename

    Attributes
    ----------
    geometry: GeometryFM2D
        Flexible Mesh geometry

    Examples
    --------
    ```{python}
    import mikeio
    mikeio.Mesh("../data/odense_rough.mesh")
    ```

    """

    def __init__(self, filename: str | Path) -> None:
        self._geometry = self._read_header(filename)
        self.plot = self.geometry.plot

    @property
    def geometry(self) -> GeometryFM2D:
        """Flexible Mesh geometry."""
        return self._geometry

    @geometry.setter
    def geometry(self, value: GeometryFM2D) -> None:
        self._geometry = value
        # plot belongs to the geometry, a new geometry needs a new plotter
        self.plot = value.plot

    def _read_header(self, filename: str | Path) -> GeometryFM2D:
        msh = MeshFile.ReadMesh(filename)

        node_table = get_nodes_from_source(msh)
        el_table = get_elements_from_source(msh)

        geom = GeometryFM2D(
            node_coordinates=node_table.coordinates,
            element_table=el_table.connectivity,
            codes=node_table.codes,
            projection=msh.ProjectionString,
            element_ids=el_table.ids,
            node_ids=node_table.ids,
            validate=False,
        )

        return geom

    def __repr__(self) -> str:
        out = [
            "<Mesh>",
            f"number of nodes: {self.n_nodes}",
            f"number of elements: {self.n_elements}",
            f"projection: {self.geometry.projection_string}",
        ]
        return str.join("\n", out)

    # TODO re-consider if all of these properties are needed, since they all are available in the geometry
    @property
    def n_elements(self) -> int:
        """Number of elements."""
        return self.geometry.n_elements

    @property
    def element_coordinates(self) -> np.ndarray:
        """Coordinates of element centroids."""
        return self.geometry.element_coordinates

    @property
    def node_coordinates(self) -> np.ndarray:
        """Coordinates of nodes (read-only, see GeometryFM2D.node_coordinates)."""
        return self.geometry.node_coordinates

    @node_coordinates.setter
    def node_coordinates(self, value: np.ndarray) -> None:
        self.geometry.node_coordinates = value

    @property
    def n_nodes(self) -> int:
        """Number of nodes."""
        return self.geometry.n_nodes

    @property
    def codes(self) -> np.ndarray:
        """Codes of nodes."""
        return self.geometry.codes

    @property
    def element_table(self) -> np.ndarray:
        """Element table."""
        return self.geometry.element_table

    @property
    def zn(self) -> np.ndarray:
        """Static bathymetry values (depth) at nodes."""
        return self.geometry.node_coordinates[:, 2]

    @zn.setter
    def zn(self, v: np.ndarray) -> None:
        if len(v) != self.n_nodes:
            raise ValueError(f"zn must have length of nodes ({self.n_nodes})")
        nc = self.geometry.node_coordinates.copy()
        nc[:, 2] = v
        self.geometry.node_coordinates = nc

    def write(
        self,
        outfilename: str | Path,
    ) -> None:
        """write mesh to file.

        Parameters
        ----------
        outfilename : str
            path to file

        """
        geometry = self.geometry

        assert isinstance(geometry, GeometryFM2D)  # i.e. not a GeometryPoint2d

        self.geometry.to_mesh(outfilename=outfilename)

    def to_shapely(self) -> MultiPolygon:
        """Convert Mesh geometry to shapely MultiPolygon.

        Returns
        -------
        MultiPolygon
            mesh as shapely MultiPolygon

        Examples
        --------
        ```{python}
        import mikeio
        msh = mikeio.open("../data/odense_rough.mesh")
        msh.to_shapely()
        ```

        """
        return self.geometry.to_shapely()
