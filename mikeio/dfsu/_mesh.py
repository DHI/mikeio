from __future__ import annotations

from mikecore.eum import eumQuantity
from mikecore.MeshBuilder import MeshBuilder
from mikecore.MeshFile import MeshFile


from ..eum import EUMType, EUMUnit
from ._dfsu import _UnstructuredFile
from  ..spatial import GeometryFM2D


class Mesh:
    """
    The Mesh class is initialized with a mesh or a dfsu file.

    Parameters
    ---------
    filename: str
        dfsu or mesh filename

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
        

    def __init__(self, filename):
        self.geometry = self._read_header(filename)
        self.plot = self.geometry.plot

    def _read_header(self, filename):
        msh = MeshFile.ReadMesh(filename)

        nc, codes, node_ids = _UnstructuredFile._get_nodes_from_source(msh)
        el_table, el_ids = _UnstructuredFile._get_elements_from_source(msh)

        geom = GeometryFM2D(
            node_coordinates=nc,
            element_table=el_table,
            codes=codes,
            projection=msh.ProjectionString,
            element_ids=el_ids,
            node_ids=node_ids,
            validate=False,
        )

        return geom

    @property
    def n_elements(self) -> int:
        """Number of elements"""
        return self.geometry.n_elements
    
    @property
    def element_coordinates(self):
        """Coordinates of element centroids"""
        return self.geometry.element_coordinates
    
    @property
    def node_coordinates(self):
        """Coordinates of nodes"""
        return self.geometry.node_coordinates

    @property
    def n_nodes(self) -> int:
        """Number of nodes"""
        return self.geometry.n_nodes
    
    @property
    def element_table(self):
        """Element table"""
        return self.geometry.element_table

    @property
    def zn(self):
        """Static bathymetry values (depth) at nodes"""
        return self.geometry.node_coordinates[:, 2]

    @zn.setter
    def zn(self, v):
        if len(v) != self.n_nodes:
            raise ValueError(f"zn must have length of nodes ({self.n_nodes})")
        self.geometry._nc[:, 2] = v
        self.geometry._ec = None

    def write(self, outfilename, elements=None, unit:EUMUnit=EUMUnit.meter):
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

        quantity = eumQuantity.Create(EUMType.Bathymetry, EUMUnit.meter)
        elem_table = _UnstructuredFile._element_table_to_mikecore(
            geometry.element_table
        )

        nc = geometry.node_coordinates
        builder.SetNodes(nc[:, 0], nc[:, 1], nc[:, 2], geometry.codes)

        builder.SetElements(elem_table)
        builder.SetProjection(geometry.projection_string)
        builder.SetEumQuantity(quantity)

        newMesh = builder.CreateMesh()
        newMesh.Write(outfilename)

    def plot_boundary_nodes(self, boundary_names=None, figsize=None, ax=None):
        return self.geometry.plot.boundary_nodes(boundary_names, figsize, ax)
