import os

from DHI.Generic.MikeZero import eumQuantity
from DHI.Generic.MikeZero.DFS.mesh import MeshFile, MeshBuilder

from ..eum import EUMType, EUMUnit
from .unstructuredbase import (
    UnstructuredType,
    _UnstructuredGeometry,
    get_nodes_from_source,
    get_elements_from_source,
)


class Mesh(_UnstructuredGeometry):

    _type = UnstructuredType.Mesh

    def __init__(self, filename):
        self._filename = filename
        if not os.path.isfile(filename):
            raise Exception(f"file {filename} does not exist!")

        self._read_header(filename)

        self._type = UnstructuredType.Mesh

    def _read_header(self, filename):
        """
        Read header of mesh file and set object properties
        """
        msh = MeshFile.ReadMesh(filename)
        self._source = msh
        self._projstr = msh.ProjectionString
        self._type = UnstructuredType.Mesh

        # geometry
        # self._set_nodes_from_source(msh)
        # self._set_elements_from_source(msh)

        self._nc, self._codes, self._n_nodes, self._node_ids = get_nodes_from_source(
            msh
        )

        (
            self._n_elements,
            self._element_table_dotnet,
            self._element_ids,
        ) = get_elements_from_source(msh)

    def __repr__(self):
        out = []
        out.append(self.type_name)
        out.append(f"Number of elements: {self.n_elements}")
        out.append(f"Number of nodes: {self.n_nodes}")
        if self._projstr:
            out.append(f"Projection: {self.projection_string}")
        return str.join("\n", out)

    @property
    def z(self):
        return self._nc[:, 2]

    @z.setter
    def z(self, value):
        """Change the depth by setting the z value of each node

        Parameters
        ----------
        z : np.array(float)
            new z value at each node
        """
        if len(value) != self.n_nodes:
            raise Exception(f"z must have length of nodes ({self.n_nodes})")
        self._nc[:, 2] = value
        self._ec = None

    def set_z(self, z):
        """Change the depth by setting the z value of each node

        Parameters
        ----------
        z : np.array(float)
            new z value at each node
        """
        if len(z) != self.n_nodes:
            raise Exception(f"z must have length of nodes ({self.n_nodes})")
        self._nc[:, 2] = z
        self._ec = None

    def set_codes(self, codes):
        """Change the code values of the nodes

        Parameters
        ----------
        codes : list(int)
            code of each node
        """
        if len(codes) != self.n_nodes:
            raise Exception(f"codes must have length of nodes ({self.n_nodes})")
        self._codes = codes
        self._valid_codes = None

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
            geometry = self
            quantity = self._source.EumQuantity
            elem_table = self._source.ElementTable
        else:
            geometry = self.elements_to_geometry(elements)
            quantity = eumQuantity.Create(EUMType.Bathymetry, EUMUnit.meter)
            elem_table = geometry._element_table_to_dotnet()

        nc = geometry.node_coordinates
        builder.SetNodes(nc[:, 0], nc[:, 1], nc[:, 2], geometry.codes)

        builder.SetElements(elem_table)
        builder.SetProjection(geometry.projection_string)
        builder.SetEumQuantity(quantity)

        newMesh = builder.CreateMesh()
        newMesh.Write(outfilename)

    def plot_boundary_nodes(self, boundary_names=None):
        """
        Plot mesh boundary nodes and their codes
        """
        import matplotlib.pyplot as plt

        nc = self.node_coordinates
        c = self.codes

        if boundary_names is not None:
            if len(self.boundary_codes) != len(boundary_names):
                raise Exception(
                    f"Number of boundary names ({len(boundary_names)}) inconsistent with number of boundaries ({len(self.boundary_codes)})"
                )
            user_defined_labels = dict(zip(self.boundary_codes, boundary_names))

        fig, ax = plt.subplots()
        for code in self.boundary_codes:
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

    @staticmethod
    def _geometry_to_mesh(outfilename, geometry):

        builder = MeshBuilder()

        nc = geometry.node_coordinates
        builder.SetNodes(nc[:, 0], nc[:, 1], nc[:, 2], geometry.codes)
        # builder.SetNodeIds(geometry.node_ids+1)
        # builder.SetElementIds(geometry.elements+1)
        builder.SetElements(geometry._element_table_to_dotnet())
        builder.SetProjection(geometry.projection_string)
        quantity = eumQuantity.Create(EUMType.Bathymetry, EUMUnit.meter)
        builder.SetEumQuantity(quantity)
        newMesh = builder.CreateMesh()
        newMesh.Write(outfilename)
