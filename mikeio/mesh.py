import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np

from DHI.Generic.MikeZero.DFS.mesh import MeshFile


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

    def plot(self, cmap=None, z=None, label=None, vmin=None, vmax=None):
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

        if vmin is None:
            vmin = z.min()

        if vmax is None:
            vmax = z.max()

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
        p.set_clim(vmin,vmax)
        ax.add_collection(p)
        fig.colorbar(p, ax=ax, label=label)
        ax.set_xlim(nc[:, 0].min(), nc[:, 0].max())
        ax.set_ylim(nc[:, 1].min(), nc[:, 1].max())

    def get_node_centered_data(self, data):
        nc = self.get_node_coords()
        ec = self.get_element_coords()
        elements = [list(self._mesh.ElementTable[i]) for i in range(len(list(self._mesh.ElementTable)))]
        elements = np.asarray(elements) - 1
        node_cellID = [list(np.argwhere(elements == i)[:, 0]) for i in np.unique(elements.reshape(-1, ))]
        node_centered_data = np.zeros(shape=nc.shape[0])
        for n, item in enumerate(node_cellID):
            InvDis = [1 / np.hypot(case[0], case[1]) for case in ec[item][:, :2] - nc[n][:2]]
            node_centered_data[n] = np.sum(InvDis * data[item]) / np.sum(InvDis)

        return node_centered_data

    def plot2(self, cmap=None, z=None, label=None, vmin=None, vmax=None, n_contour=10):
        """
        Plot mesh elements and contour

        Parameters
        ----------
        cmap: matplotlib.cm.cmap, optional
            default viridis
        z: np.array
            value for each element to plot, default bathymetry
        label: str, optional
            colorbar label
        vmin: float, optional
            minimum value to show on colorbar
        vmax: float, optional
            maximum value to show on colorbar
        n_contour, int, optional
            number of contour levels to plot
        """
        import matplotlib.tri as tri
        if cmap is None:
            cmap = cm.viridis

        nc = self.get_node_coords()
        ec = self.get_element_coords()

        elements = [list(self._mesh.ElementTable[i]) for i in range(len(list(self._mesh.ElementTable)))]
        elements = np.asarray(elements) - 1

        if z is None:
            zz = nc[:, 2]
            if label is None:
                label = "Bathymetry (m)"
        else:
            zz = self.get_node_centered_data(z)

        if vmin is None:
            vmin = zz.min()

        if vmax is None:
            vmax = zz.max()

        triang = tri.Triangulation(nc[:, 0], nc[:, 1], elements)
        refiner = tri.UniformTriRefiner(triang)
        tri_refi, z_test_refi = refiner.refine_field(zz, subdiv=3)

        fig, ax = plt.subplots()
        ax.triplot(triang, lw=0.5, color='blue')
        tr_fig = ax.tripcolor(tri_refi, z_test_refi, edgecolors='face', vmin=vmin, vmax=vmax)
        levels = np.linspace(vmin, vmax, n_contour)
        ax.tricontourf(tri_refi, z_test_refi, levels=levels, cmap=cmap)
        ax.tricontour(tri_refi, z_test_refi, levels=levels,
                      colors=['0.5'],
                      linewidths=[0.5])
        plt.colorbar(tr_fig, label=label)
