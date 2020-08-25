import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np

from DHI.Generic.MikeZero import eumQuantity
from DHI.Generic.MikeZero.DFS.mesh import MeshFile, MeshBuilder

from .dfsu import _UnstructuredGeometry
from .eum import EUMType, EUMUnit
from .dotnet import asnetarray_v2

class Mesh:
    def __init__(self, filename):
        self._mesh = MeshFile.ReadMesh(filename)

    def get_number_of_elements(self):
        return self._mesh.NumberOfElements

    @property
    def number_of_elements(self):
        return self._mesh.NumberOfElements

    @property
    def number_of_nodes(self):
        return self._mesh.NumberOfNodes

    def get_number_of_nodes(self):
        return self._mesh.NumberOfNodes

    def get_node_coords(self, code=None):
        """
        Get node coordinates
        
        Parameters
        ----------
        code: int, optional
            filter results by code, land==1

        Returns
        -------
        np.ndarray
        """
        # Node coordinates
        xn = np.array(list(self._mesh.X))
        yn = np.array(list(self._mesh.Y))
        zn = np.array(list(self._mesh.Z))

        nc = np.column_stack([xn, yn, zn])

        if code is not None:

            c = np.array(list(self._mesh.Code))
            valid_codes = set(c)

            if code not in valid_codes:

                print(f"Selected code: {code} is not valid. Valid codes: {valid_codes}")
                raise Exception
            return nc[c == code]

        return nc

    def get_element_coords(self):
        """ Calculate element center coordinates

            Returns
                np.array((number_of_elements, 3)
        """
        ne = self._mesh.NumberOfElements

        # Node coordinates
        xn = np.array(list(self._mesh.X))
        yn = np.array(list(self._mesh.Y))
        zn = np.array(list(self._mesh.Z))

        ec = np.empty([ne, 3])

        for j in range(ne):
            nodes = self._mesh.ElementTable[j]

            xcoords = np.empty(nodes.Length)
            ycoords = np.empty(nodes.Length)
            zcoords = np.empty(nodes.Length)
            for i in range(nodes.Length):
                nidx = nodes[i] - 1
                xcoords[i] = xn[nidx]
                ycoords[i] = yn[nidx]
                zcoords[i] = zn[nidx]

            ec[j, 0] = xcoords.mean()
            ec[j, 1] = ycoords.mean()
            ec[j, 2] = zcoords.mean()

        return ec

    def to_shapely(self):
        from shapely.geometry import Polygon, MultiPolygon

        nc = self.get_node_coords()

        ne = self.get_number_of_elements()

        polys = []
        for j in range(ne):
            nodes = self._mesh.ElementTable[j]
            pcoords = np.empty([nodes.Length, 2])
            for i in range(nodes.Length):
                nidx = nodes[i] - 1
                pcoords[i, :] = nc[nidx, 0:2]
            polygon = Polygon(pcoords)
            polys.append(polygon)
        mp = MultiPolygon(polys)

        return mp

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

        nc = self.get_node_coords()
        ec = self.get_element_coords()
        ne = ec.shape[0]

        if z is None:
            z = ec[:, 2]
            if label is None:
                label = "Bathymetry (m)"

        patches = []

        for j in range(ne):
            nodes = self._mesh.ElementTable[j]
            pcoords = np.empty([nodes.Length, 2])
            for i in range(nodes.Length):
                nidx = nodes[i] - 1
                pcoords[i, :] = nc[nidx, 0:2]

            polygon = Polygon(pcoords, True)
            patches.append(polygon)

        fig, ax = plt.subplots()
        p = PatchCollection(patches, cmap=cmap, edgecolor="black")

        p.set_array(z)
        ax.add_collection(p)
        fig.colorbar(p, ax=ax, label=label)
        ax.set_xlim(nc[:, 0].min(), nc[:, 0].max())
        ax.set_ylim(nc[:, 1].min(), nc[:, 1].max())

    def write(self, outfilename):
        projection = self._mesh.ProjectionString
        eumQuantity = self._mesh.EumQuantity

        builder = MeshBuilder()

        nc = self.get_node_coords()

        x = self._mesh.X
        y = self._mesh.Y
        z = self._mesh.Z
        c = self._mesh.Code

        builder.SetNodes(x,y,z,c)
        builder.SetElements(self._mesh.ElementTable)
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

        elem_table = asnetarray_v2(geometry.element_table)

        builder.SetNodes(x,y,z,c)
        builder.SetNodeIds(geometry.node_ids)
        builder.SetElementIds(geometry.element_ids)
        builder.SetElements(elem_table)
        builder.SetProjection(projection)
        builder.SetEumQuantity(quantity)
        newMesh = builder.CreateMesh()
        newMesh.Write(outfilename)
