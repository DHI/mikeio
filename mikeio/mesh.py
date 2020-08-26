import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np

from DHI.Generic.MikeZero import eumQuantity
from DHI.Generic.MikeZero.DFS.mesh import MeshFile, MeshBuilder

from .dfsu import _UnstructuredGeometry, _UnstructuredFile
from .eum import EUMType, EUMUnit
from .dotnet import asnetarray_v2

class Mesh(_UnstructuredFile):
    def __init__(self, filename):
        #self._mesh = MeshFile.ReadMesh(filename)
        super().__init__()
        self._filename = filename
        self._read_mesh_header(filename)

    def plot(self, cmap=None, z=None, label=None):
        """
        Plot mesh elements

        Parameters
        ----------
        cmap: matplotlib.cm.cmap, optional
            default viridis
        z: np.array
            value for each element to plot, default bathymetry
        label: str, optional
            colorbar label
        """
        if cmap is None:
            cmap = cm.viridis

        nc = self.node_coordinates
        ec = self.element_coordinates
        ne = ec.shape[0]

        if z is None:
            z = ec[:, 2]
            if label is None:
                label = "Bathymetry (m)"

        # patches = []
        # for j in range(ne):
        #     nodes = self._mesh.ElementTable[j]
        #     pcoords = np.empty([nodes.Length, 2])
        #     for i in range(nodes.Length):
        #         nidx = nodes[i] - 1
        #         pcoords[i, :] = nc[nidx, 0:2]

        #     polygon = Polygon(pcoords, True)
        #     patches.append(polygon)

        fig, ax = plt.subplots()
        patches = self.to_polygons()
        #p = PatchCollection(patches, cmap=cmap, edgecolor="black")
        p = PatchCollection(patches, cmap=cmap, edgecolor="lightgray", alpha=0.2)

        p.set_array(z)
        ax.add_collection(p)
        fig.colorbar(p, ax=ax, label=label)
        ax.set_xlim(nc[:, 0].min(), nc[:, 0].max())
        ax.set_ylim(nc[:, 1].min(), nc[:, 1].max())

    def write(self, outfilename):
        projection = self._source.ProjectionString
        eumQuantity = self._source.EumQuantity
        # TODO: use member properties instead of using _source
        
        builder = MeshBuilder()

        nc = self.node_coordinates

        x = self._source.X
        y = self._source.Y
        z = self._source.Z
        c = self._source.Code

        builder.SetNodes(x,y,z,c)
        builder.SetElements(self._source.ElementTable)
        builder.SetProjection(projection)
        builder.SetEumQuantity(eumQuantity)
        newMesh = builder.CreateMesh()
        newMesh.Write(outfilename)

    @staticmethod
    def geometry_to_mesh(outfilename, geometry):
        projection = geometry.projection_string        
        quantity = eumQuantity.Create(EUMType.Bathymetry, EUMUnit.meter)

        builder = MeshBuilder()

        nc = geometry.node_coordinates

        x = nc[:,0]
        y = nc[:,1]
        z = nc[:,2]
        c = geometry.codes

        elem_table = []
        for j in range(geometry.n_elements):
            elem_nodes = geometry.element_table[j]
            elem_nodes = [nd+1 for nd in elem_nodes]  
            elem_table.append(elem_nodes)
        elem_table = asnetarray_v2(elem_table)

        builder.SetNodes(x,y,z,c)
        #builder.SetNodeIds(geometry.node_ids)
        #builder.SetElementIds(geometry.element_ids)
        builder.SetElements(elem_table)
        builder.SetProjection(projection)
        builder.SetEumQuantity(quantity)
        newMesh = builder.CreateMesh()
        newMesh.Write(outfilename)
