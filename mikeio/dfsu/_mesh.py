from __future__ import annotations
from pathlib import Path
from collections.abc import Collection
import warnings


import numpy as np


from mikecore.eum import eumQuantity
from mikecore.MeshBuilder import MeshBuilder
from mikecore.MeshFile import MeshFile

from ..eum import EUMType, EUMUnit
from ..spatial import GeometryFM2D
from ._common import (
    get_elements_from_source,
    get_nodes_from_source,
    element_table_to_mikecore,
)


class Mesh:
    """
    The Mesh class is initialized with a mesh file.

    Parameters
    ---------
    filename: str
        mesh filename

    Examples
    --------
    >>> import mikeio
    >>> msh = mikeio.Mesh("tests/testdata/odense_rough.mesh")
    >>> msh
    Flexible Mesh
    number of elements: 654
    number of nodes: 399
    projection: UTM-33
    """

    def __init__(self, filename: str | Path) -> None:
        # Mesh used to be able to read dfsu files, but not anymore
        ext = Path(filename).suffix.lower()

        if ext == ".dfsu":
            import mikeio

            warnings.warn(
                f'Reading dfsu with `Mesh` is deprecated. Read a .dfsu geometry with `geom = mikeio.open("{filename}").geometry`',
                FutureWarning,
            )
            self.geometry = mikeio.open(str(filename)).geometry
        elif ext == ".mesh":
            self.geometry = self._read_header(filename)

        self.plot = self.geometry.plot

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
            "Flexible Mesh",
            f"number of elements: {self.n_elements}",
            f"number of nodes: {self.n_nodes}",
            f"projection: {self.geometry.projection_string}",
        ]
        return str.join("\n", out)

    @property
    def n_elements(self) -> int:
        """Number of elements"""
        return self.geometry.n_elements

    @property
    def element_coordinates(self) -> np.ndarray:
        """Coordinates of element centroids"""
        return self.geometry.element_coordinates

    @property
    def node_coordinates(self) -> np.ndarray:
        """Coordinates of nodes"""
        return self.geometry.node_coordinates

    @property
    def n_nodes(self) -> int:
        """Number of nodes"""
        return self.geometry.n_nodes

    @property
    def element_table(self) -> np.ndarray:
        """Element table"""
        return self.geometry.element_table

    @property
    def zn(self) -> np.ndarray:
        """Static bathymetry values (depth) at nodes"""
        return self.geometry.node_coordinates[:, 2]

    @zn.setter
    def zn(self, v: np.ndarray):
        if len(v) != self.n_nodes:
            raise ValueError(f"zn must have length of nodes ({self.n_nodes})")
        self.geometry._nc[:, 2] = v

    def write(
        self,
        outfilename: str | Path,
        elements: Collection[int] | None = None,
        unit: EUMUnit = EUMUnit.meter,
    ) -> None:
        """write mesh to file (will overwrite if file exists)

        Parameters
        ----------
        outfilename : str
            path to file
        elements : list(int)
            list of element ids (subset) to be saved to new mesh
        """
        builder = MeshBuilder()

        if elements is not None:
            geometry = self.geometry.isel(elements)
        else:
            geometry = self.geometry

        assert isinstance(geometry, GeometryFM2D)  # i.e. not a GeometryPoint2d

        quantity = eumQuantity.Create(EUMType.Bathymetry, unit)
        elem_table = element_table_to_mikecore(geometry.element_table)

        nc = geometry.node_coordinates
        builder.SetNodes(nc[:, 0], nc[:, 1], nc[:, 2], geometry.codes)

        builder.SetElements(elem_table)
        builder.SetProjection(geometry.projection_string)
        builder.SetEumQuantity(quantity)

        newMesh = builder.CreateMesh()
        newMesh.Write(outfilename)

    def plot_boundary_nodes(self, boundary_names=None, figsize=None, ax=None) -> None:
        return self.geometry.plot.boundary_nodes(boundary_names, figsize, ax)
