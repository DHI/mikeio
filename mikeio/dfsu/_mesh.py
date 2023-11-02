from __future__ import annotations

from mikecore.eum import eumQuantity
from mikecore.MeshBuilder import MeshBuilder


from ..eum import EUMType, EUMUnit

from ._dfsu import _UnstructuredFile


class Mesh(_UnstructuredFile):
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
        super().__init__()
        self._filename = filename
        self._read_header(filename)
        self._n_timesteps = None
        self._type = None  # DfsuFileType.Mesh
        self.plot = self.geometry.plot

    @property
    def zn(self):
        """Static bathymetry values (depth) at nodes"""
        return self.node_coordinates[:, 2]

    @zn.setter
    def zn(self, v):
        if len(v) != self.n_nodes:
            raise ValueError(f"zn must have length of nodes ({self.n_nodes})")
        self._geometry._nc[:, 2] = v
        self._geometry._ec = None

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
            geometry = self.geometry
            if hasattr(self._source, "EumQuantity"):
                quantity = self._source.EumQuantity
            else:
                quantity = eumQuantity.Create(EUMType.Bathymetry, self._source.ZUnit)
            elem_table = self._source.ElementTable
        else:
            geometry = self.geometry.elements_to_geometry(elements)
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
