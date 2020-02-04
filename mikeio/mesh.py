import numpy as np
from DHI.Generic.MikeZero.DFS.mesh import MeshFile


class mesh:
    def read(self, filename):
        """ Function: Read a mesh file

        usage:
            read(filename)

        Returns
            Nothing
        """
        self._mesh = MeshFile.ReadMesh(filename)

    def get_number_of_elements(self):
        return self._mesh.NumberOfElements

    def get_number_of_nodes(self):
        return self._mesh.NumberOfNodes

    def get_node_coords(self, code=None):
        """
        Function: get node coordinates, optionally filtered by code, land==1
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
        """ Function: Calculate element center coordinates

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
