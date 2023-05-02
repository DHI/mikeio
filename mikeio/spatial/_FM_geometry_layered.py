import warnings
from collections import namedtuple
from functools import cached_property
from typing import Collection, Sequence, Union, Optional

import numpy as np
from mikecore.DfsuFile import DfsuFileType  # type: ignore
from mikecore.eum import eumQuantity  # type: ignore
from mikecore.MeshBuilder import MeshBuilder  # type: ignore
from scipy.spatial import cKDTree

from ._FM_geometry import GeometryFM2D
from ._geometry import _Geometry

from ._FM_utils import _plot_vertical_profile

from ._utils import _relative_cumulative_distance


class _GeometryFMLayered(_Geometry):
    def __init__(
        self,
        *,
        node_coordinates,
        element_table,
        codes=None,
        projection=None,
        dfsu_type=DfsuFileType.Dfsu3DSigma,
        element_ids=None,
        node_ids=None,
        n_layers: int = 1,  # at least 1 layer
        n_sigma=None,
        validate=True,
    ) -> None:

        super().__init__(projection=projection)

        # super().__init__(
        #    node_coordinates=node_coordinates,
        #    element_table=element_table,
        #    codes=codes,
        #    projection=projection,
        #    dfsu_type=dfsu_type,
        #    element_ids=element_ids,
        #    node_ids=node_ids,
        #    validate=validate,
        # )

        self._type = dfsu_type

        self._n_layers_column = None  # lazy
        self._bot_elems = None  # lazy
        self._n_layers = n_layers
        self._n_sigma = n_sigma

        self.node_coordinates = np.asarray(node_coordinates)
        n_nodes = len(node_coordinates)
        self.element_table = element_table  # 3d
        self.codes = (
            np.zeros((n_nodes,), dtype=int) if codes is None else np.asarray(codes)
        )

        # TODO remove this
        self._geometry = GeometryFM2D(
            node_coordinates=node_coordinates,
            element_table=element_table,
            codes=codes,
            projection=projection,
            dfsu_type=dfsu_type,
            element_ids=element_ids,
            node_ids=node_ids,
            validate=validate,
        )

        self._geometry2d = self.to_2d_geometry()

        # TODO remove
        self._geometry._geometry2d = self._geometry2d

        self._e2_e3_table = None  # lazy
        self._2d_ids = None  # lazy
        self._layer_ids = None  # lazy
        # self.__dz = None  # lazy

    @property
    def geometry2d(self):
        """The 2d geometry for a 3d object"""
        return self._geometry2d

    def _get_nodes_and_table_for_elements(self, elements, node_layers="all"):
        """list of nodes and element table for a list of elements

        Parameters
        ----------
        elements : np.array(int)
            array of element ids
        node_layers : str, optional
            for 3D files 'all', 'bottom' or 'top' nodes
            of each element, by default 'all'

        Returns
        -------
        np.array(int)
            array of node ids (unique)
        list(list(int))
            element table with a list of nodes for each element
        """
        elem_tbl = np.empty(len(elements), dtype=np.dtype("O"))
        if (node_layers is None) or (node_layers == "all") or self.is_2d:
            for j, eid in enumerate(elements):
                elem_tbl[j] = np.asarray(self.element_table[eid])

        else:
            # 3D => 2D
            if (node_layers != "bottom") and (node_layers != "top"):
                raise Exception("node_layers must be either all, bottom or top")
            for j, eid in enumerate(elements):
                elem_nodes = np.asarray(self.element_table[eid])
                nn = len(elem_nodes)
                halfn = int(nn / 2)
                if node_layers == "bottom":
                    elem_nodes = elem_nodes[:halfn]
                if node_layers == "top":
                    elem_nodes = elem_nodes[halfn:]
                elem_tbl[j] = elem_nodes

        nodes = np.unique(np.hstack(elem_tbl))
        return nodes, elem_tbl

    def to_2d_geometry(self):
        """extract 2d geometry from 3d geometry

        Returns
        -------
        UnstructuredGeometry
            2d geometry (bottom nodes)
        """
        # extract information for selected elements
        elem_ids = self.bottom_elements
        if self._type == DfsuFileType.Dfsu3DSigmaZ:
            # for z-layers nodes will not match on neighboring elements!
            elem_ids = self.top_elements

        node_ids, elem_tbl = self._get_nodes_and_table_for_elements(
            elem_ids, node_layers="bottom"
        )
        node_coords = self.node_coordinates[node_ids]
        codes = self.codes[node_ids]

        # create new geometry
        geom = GeometryFM2D(
            node_coordinates=node_coords,
            codes=codes,
            node_ids=node_ids,
            projection=self.projection_string,
            element_table=elem_tbl,
            element_ids=self.element_ids[elem_ids],
            validate=False,
            dfsu_type=DfsuFileType.Dfsu2D,
        )

        geom._type = None  # DfsuFileType.Mesh
        geom._reindex()

        # Fix z-coordinate for sigma-z:
        if self._type == DfsuFileType.Dfsu3DSigmaZ:
            zn = geom.node_coordinates[:, 2].copy()
            for j, elem_nodes in enumerate(geom.element_table):
                elem_nodes3d = self.element_table[self.bottom_elements[j]]
                for jn in range(len(elem_nodes)):
                    znj_3d = self.node_coordinates[elem_nodes3d[jn], 2]
                    zn[elem_nodes[jn]] = min(zn[elem_nodes[jn]], znj_3d)
            geom.node_coordinates[:, 2] = zn

        return geom

    @property
    def n_elements(self):
        """Number of 3d elements"""
        return self._geometry.n_elements

    @property
    def element_ids(self):
        """Element ids"""
        return self._geometry.element_ids

    @property
    def is_2d(self):
        return False

    @property
    def ndim(self) -> int:
        return 3

    @cached_property
    def layer_ids(self):
        """The layer number (0=bottom, 1, 2, ...) for each 3d element"""
        if self._layer_ids is None:
            res = self._get_2d_to_3d_association()
            self._e2_e3_table = res[0]
            self._2d_ids = res[1]
            self._layer_ids = res[2]
        return self._layer_ids

    @property
    def n_layers(self):
        """Maximum number of layers"""
        return self._n_layers

    @property
    def n_sigma_layers(self):
        """Number of sigma layers"""
        return self._n_sigma

    @property
    def n_z_layers(self):
        """Maximum number of z-layers"""
        return self.n_layers - self.n_sigma_layers

    @cached_property
    def top_elements(self):
        """List of 3d element ids of surface layer"""
        # note: if subset of elements is selected then this cannot be done!

        # fast path if no z-layers
        if self.n_z_layers == 0:
            return np.arange(
                start=self.n_sigma_layers - 1,
                stop=self.n_elements,
                step=self.n_sigma_layers,
            )
        else:
            # slow path
            return self._findTopLayerElements(self._geometry.element_table)

    def find_index(self, x=None, y=None, z=None, coords=None, area=None, layers=None):

        if layers is not None:
            idx = self.get_layer_elements(layers)
        else:
            idx = self.element_ids

        if (
            (coords is not None)
            or (x is not None)
            or (y is not None)
            or (z is not None)
        ):
            if area is not None:
                raise ValueError(
                    "Coordinates and area cannot be provided at the same time!"
                )
            if coords is not None:
                coords = np.atleast_2d(coords)
                xy = coords[:, :2]
                z = coords[:, 2] if coords.shape[1] == 3 else None
            else:
                xy = np.vstack((x, y)).T
            idx_2d = self._geometry2d._find_element_2d(coords=xy)
            assert len(idx_2d) == len(xy)
            if z is None:
                idx_3d = np.hstack(self.e2_e3_table[idx_2d])
            else:
                idx_3d = self._find_elem3d_from_elem2d(idx_2d, z)
            idx = np.intersect1d(idx, idx_3d).astype(int)
        elif area is not None:
            idx_area = self._geometry._elements_in_area(area)
            idx = np.intersect1d(idx, idx_area)
        elif layers is None:
            raise ValueError(
                "At least one selection argument (x,y,z,coords,area,layers) needs to be provided!"
            )
        return idx

    @staticmethod
    def _findTopLayerElements(elementTable):
        """
        Find element indices (zero based) of the elements being the upper-most element
        in its column.
        Each column is identified by matching node id numbers. For 3D elements the
        last half of the node numbers of the bottom element must match the first half
        of the node numbers in the top element. For 2D vertical elements the order of
        the node numbers in the bottom element (last half number of nodes) are reversed
        compared to those in the top element (first half number of nodes).
        To find the number of elements in each column, assuming the result
        is stored in res:
        For the first column it is res[0]+1.
        For the i'th column, it is res[i]-res[i-1].
        :returns: A list of element indices of top layer elements
        """

        topLayerElments = []

        # Find top layer elements by matching the number numers of the last half of elmt i
        # with the first half of element i+1.
        # Elements always start from the bottom, and the element of one columne are following
        # each other in the element table.
        for i in range(len(elementTable) - 1):
            elmt1 = elementTable[i]
            elmt2 = elementTable[i + 1]

            if len(elmt1) != len(elmt2):
                # elements with different number of nodes can not be on top of each other,
                # so elmt2 must be another column, and elmt1 must be a top element
                topLayerElments.append(i)
                continue

            if len(elmt1) % 2 != 0:
                raise Exception(
                    "In a layered mesh, each element must have an even number of elements (element index "
                    + i
                    + ")"
                )

            # Number of nodes in a 2D element
            elmt2DSize = len(elmt1) // 2

            for j in range(elmt2DSize):
                if elmt2DSize > 2:
                    if elmt1[j + elmt2DSize] != elmt2[j]:
                        # At least one node number did not match
                        # so elmt2 must be another column, and elmt1 must be a top element
                        topLayerElments.append(i)
                        break
                else:
                    # for 2D vertical profiles the nodes in the element on the
                    # top is in reverse order of those in the bottom.
                    if elmt1[j + elmt2DSize] != elmt2[(elmt2DSize - 1) - j]:
                        # At least one node number did not match
                        # so elmt2 must be another column, and elmt1 must be a top element
                        topLayerElments.append(i)
                        break

        # The last element will always be a top layer element
        topLayerElments.append(len(elementTable) - 1)

        return np.array(topLayerElments, dtype=np.int32)

    @property
    def n_layers_per_column(self):
        """List of number of layers for each column"""
        if self._n_layers is None:
            print("Object has no layers: cannot find n_layers_per_column")
            return None
        elif self._n_layers_column is None:
            top_elems = self.top_elements
            n = len(top_elems)
            tmp = top_elems.copy()
            tmp[0] = -1
            tmp[1:n] = top_elems[0 : (n - 1)]
            self._n_layers_column = top_elems - tmp
        return self._n_layers_column

    @cached_property
    def bottom_elements(self):
        """List of 3d element ids of bottom layer"""
        return self.top_elements - self.n_layers_per_column + 1

    def get_layer_elements(self, layers, layer=None):
        """3d element ids for one (or more) specific layer(s)

        Parameters
        ----------
        layers : int or list(int)
            layer between 0 (bottom) and n_layers-1 (top)
            (can also be negative counting from -1 at the top layer)

        Returns
        -------
        np.array(int)
            element ids
        """
        if layer is not None:
            warnings.warn(
                "layer argument is deprecated, use layers instead",
                FutureWarning,
            )
            layers = layer

        if isinstance(layers, str):
            if layers in ("surface", "top"):
                return self.top_elements
            elif layers in ("bottom"):
                return self.bottom_elements
            else:
                raise ValueError(
                    f"layers '{layers}' not recognized ('top', 'bottom' or integer)"
                )

        if not np.isscalar(layers):
            elem_ids = []
            for layer in layers:
                elem_ids.append(self.get_layer_elements(layer))
            elem_ids = np.concatenate(elem_ids, axis=0)
            return np.sort(elem_ids)

        n_lay = self.n_layers
        if n_lay is None:
            raise InvalidGeometry("Object has no layers: cannot get_layer_elements")

        if layers < (-n_lay) or layers >= n_lay:
            raise Exception(
                f"Layer {layers} not allowed; must be between -{n_lay} and {n_lay-1}"
            )

        if layers < 0:
            layers = layers + n_lay

        return self.element_ids[self.layer_ids == layers]

    @property
    def e2_e3_table(self):
        """The 2d-to-3d element connectivity table for a 3d object"""

        # e2_e3, 2d_ids and layer_ids are all set at the same time

        if self._e2_e3_table is None:
            res = self._get_2d_to_3d_association()
            self._e2_e3_table = res[0]
            self._2d_ids = res[1]
            self._layer_ids = res[2]
        return self._e2_e3_table

    @property
    def elem2d_ids(self):
        """The associated 2d element id for each 3d element"""
        if self._2d_ids is None:
            res = self._get_2d_to_3d_association()
            self._e2_e3_table = res[0]
            self._2d_ids = res[1]
            self._layer_ids = res[2]
        return self._2d_ids

    def _get_2d_to_3d_association(self):
        e2_to_e3 = (
            []
        )  # for each 2d element: the corresponding 3d element ids from bot to top
        index2d = []  # for each 3d element: the associated 2d element id
        layerid = []  # for each 3d element: the associated layer number
        n2d = len(self.top_elements)
        topid = self.top_elements
        botid = self.bottom_elements
        # layer_ids = 0, 1, 2...
        global_layer_ids = np.arange(self.n_layers)
        for j in range(n2d):
            col = np.arange(botid[j], topid[j] + 1)

            e2_to_e3.append(col)
            for _ in col:
                index2d.append(j)

            n_local_layers = len(col)
            local_layers = global_layer_ids[-n_local_layers:]
            for ll in local_layers:
                layerid.append(ll)

        e2_to_e3 = np.array(e2_to_e3, dtype=object)
        index2d = np.array(index2d)
        layerid = np.array(layerid)
        return e2_to_e3, index2d, layerid

    def _find_elem3d_from_elem2d(self, elem2d, z):
        """Find 3d element ids from 2d element ids and z-values"""

        # TODO: coordinate with _find_3d_from_2d_points()

        elem2d = [elem2d] if np.isscalar(elem2d) else elem2d
        elem2d = np.asarray(elem2d)
        z_vec = np.full(elem2d.shape, fill_value=z) if np.isscalar(z) else z
        elem3d = np.full_like(elem2d, fill_value=-1)
        for j, e2 in enumerate(elem2d):
            idx_3d = np.hstack(self.e2_e3_table[e2])
            elem3d[j] = idx_3d[self._z_idx_in_column(idx_3d, z_vec[j])]

            # z_col = self.element_coordinates[idx_3d, 2]
            # elem3d[j] = (np.abs(z_col - z_vec[j])).argmin()  # nearest
        return elem3d

    def _find_3d_from_2d_points(self, elem2d, z=None, layer=None):

        was_scalar = np.isscalar(elem2d)
        if was_scalar:
            elem2d = np.array([elem2d])
        else:
            orig_shape = elem2d.shape
            elem2d = np.reshape(elem2d, (elem2d.size,))

        if (layer is None) and (z is None):
            # return top element
            idx = self.top_elements[elem2d]  # TODO: return whole column instead

        elif layer is None:
            idx = np.zeros_like(elem2d)
            if np.isscalar(z):
                z = z * np.ones_like(elem2d, dtype=float)
            elem3d = self.e2_e3_table[elem2d]
            for j, row in enumerate(elem3d):
                zc = self.element_coordinates[row, 2]
                d3d = np.abs(z[j] - zc)
                idx[j] = row[d3d.argsort()[0]]

        elif z is None:
            if 0 <= layer <= self.n_z_layers - 1:
                idx = np.zeros_like(elem2d)
                elem3d = self.e2_e3_table[elem2d]
                for j, row in enumerate(elem3d):
                    try:
                        layer_ids = self.layer_ids[row]
                        id = row[list(layer_ids).index(layer)]
                        idx[j] = id
                    except:
                        print(f"Layer {layer} not present for 2d element {elem2d[j]}")
            else:
                # sigma layer
                idx = self.get_layer_elements(layer)[elem2d]

        else:
            raise ValueError("layer and z cannot both be supplied!")

        if was_scalar:
            idx = idx[0]
        else:
            idx = np.reshape(idx, orig_shape)

        return idx

    def calc_element_coordinates(self, elements=None, zn=None):
        """Calculates the coordinates of the center of each element.

        Only necessary for dynamic vertical coordinates,
        otherwise use the property *element_coordinates* instead

        Parameters
        ----------
        elements : np.array(int), optional
            element ids of selected elements
        zn : np.array(float), optional
            only the z-coodinates of the nodes

        Examples
        --------

        Returns
        -------
        np.array
            x,y,z of each element
        """
        return self._calc_element_coordinates(elements, zn)

    @cached_property
    def _dz(self):
        """Height of each 3d element (using static zn information)"""
        return self._calc_dz()

    def _calc_dz(self, elements=None, zn=None):
        """Height of 3d elements using static or dynamic zn information"""
        if elements is None:
            element_table = self.element_table
        else:
            element_table = self.element_table[elements]
        n_elements = len(element_table)

        if zn is None:
            if elements is None:
                zn = self.node_coordinates[:, 2]
            else:
                nodes = np.unique(np.hstack(element_table))
                zn = self.node_coordinates[nodes, 2]

        zn_is_2d = len(zn.shape) == 2
        shape = (zn.shape[0], n_elements) if zn_is_2d else (n_elements,)
        dz = np.full(shape=shape, fill_value=np.nan)

        if zn_is_2d:
            # dynamic zn
            for j in range(n_elements):
                nodes = element_table[j]
                halfn = int(len(nodes) / 2)
                z_bot = np.mean(zn[:, nodes[:halfn]], axis=1)
                z_top = np.mean(zn[:, nodes[halfn:]], axis=1)
                dz[:, j] = z_top - z_bot
        else:
            # static zn
            for j in range(n_elements):
                nodes = element_table[j]
                halfn = int(len(nodes) / 2)
                z_bot = np.mean(zn[nodes[:halfn]])
                z_top = np.mean(zn[nodes[halfn:]])
                dz[j] = z_top - z_bot

        return dz

    # TODO: add methods for extracting layers etc


class GeometryFM3D(_GeometryFMLayered):
    pass


class GeometryFMVerticalProfile(_GeometryFMLayered):
    def __init__(
        self,
        node_coordinates,
        element_table,
        codes=None,
        projection=None,
        dfsu_type=None,
        element_ids=None,
        node_ids=None,
        n_layers: int = 1,
        n_sigma=None,
        validate=True,
    ) -> None:
        super().__init__(
            node_coordinates=node_coordinates,
            element_table=element_table,
            codes=codes,
            projection=projection,
            dfsu_type=dfsu_type,
            element_ids=element_ids,
            node_ids=node_ids,
            n_layers=n_layers,
            n_sigma=n_sigma,
            validate=validate,
        )
        self.plot = _GeometryFMVerticalProfilePlotter(self)
        # self._rel_node_dist = None
        self._rel_elem_dist = None

        # remove inherited but unsupported methods
        def __dir__(self):
            natdir = set(self.__dict__.keys() + dir(self.__class__))
            natdir.remove("interp2d")
            return list(natdir)

        # self.interp2d = None

    @property
    def boundary_polylines(self):
        # Overides base class
        raise AttributeError(
            "GeometryFMVerticalProfile has no boundary_polylines property"
        )

    # remove unsupported methods from dir to avoid confusion
    def __dir__(self):
        unsupported = ("boundary_polylines", "interp2d")
        keys = sorted(list(super().__dir__()) + list(self.__dict__.keys()))
        return set([d for d in keys if d not in unsupported])

    # @property
    # def relative_node_distance(self):
    #     if self._rel_node_dist is None:
    #         nc = self.node_coordinates
    #         self._rel_node_dist = _relative_cumulative_distance(nc, is_geo=self.is_geo)
    #     return self._rel_node_dist

    @property
    def relative_element_distance(self):
        if self._rel_elem_dist is None:
            ec = self.element_coordinates
            nc0 = self.node_coordinates[0, :2]
            self._rel_elem_dist = _relative_cumulative_distance(
                ec, nc0, is_geo=self.is_geo
            )
        return self._rel_elem_dist

    def get_nearest_relative_distance(self, coords) -> float:
        """For a point near a transect, find the nearest relative distance
        for showing position on transect plot.

        Parameters
        ----------
        coords : [float, float]
            x,y-coordinate of point

        Returns
        -------
        float
            relative distance in meters from start of transect
        """
        xe = self.element_coordinates[:, 0]
        ye = self.element_coordinates[:, 1]
        dd2 = np.square(xe - coords[0]) + np.square(ye - coords[1])
        idx = np.argmin(dd2)
        return self.relative_element_distance[idx]

    def find_index(self, x=None, y=None, z=None, coords=None, layers=None):

        if layers is not None:
            idx = self.get_layer_elements(layers)
        else:
            idx = self.element_ids

        # select in space
        if (
            (coords is not None)
            or (x is not None)
            or (y is not None)
            or (z is not None)
        ):
            if coords is not None:
                coords = np.atleast_2d(coords)
                xy = coords[:, :2]
                z = coords[:, 2] if coords.shape[1] == 3 else None
            else:
                xy = np.vstack((x, y)).T

            idx_2d = self._find_nearest_element_2d(coords=xy)

            if z is None:
                idx_3d = np.hstack(self.e2_e3_table[idx_2d])
            else:
                idx_3d = self._find_elem3d_from_elem2d(idx_2d, z)
            idx = np.intersect1d(idx, idx_3d)

        return idx

    def _find_nearest_element_2d(self, coords):
        ec2d = self.element_coordinates[self.top_elements, :2]
        xe, ye = ec2d[:, 0], ec2d[:, 1]
        coords = np.atleast_2d(coords)
        idx = np.zeros(len(coords), dtype=int)
        for j, pt in enumerate(coords):
            x, y = pt[0:2]
            idx[j] = np.argmin((xe - x) ** 2 + (ye - y) ** 2)
        return idx


class GeometryFMVerticalColumn(GeometryFM3D):
    def calc_ze(self, zn=None):
        if zn is None:
            zn = self.node_coordinates[:, 2]
        return self._calc_z_using_idx(zn, self._idx_e)

    def calc_zf(self, zn=None):
        if zn is None:
            zn = self.node_coordinates[:, 2]
        return self._calc_z_using_idx(zn, self._idx_f)

    def _calc_zee(self, zn=None):
        ze = self.calc_ze(zn)
        zf = self.calc_zf(zn)
        if ze.ndim == 1:
            zee = np.insert(ze, 0, zf[0])
            return np.append(zee, zf[-1])
        else:
            return np.hstack(
                (zf[:, 0].reshape((-1, 1)), ze, zf[:, -1].reshape((-1, 1)))
            )

    def _interp_values(self, zn, data, z):
        """Interpolate to other z values, allow linear extrapolation"""
        from scipy.interpolate import interp1d  # type: ignore

        opt = {"kind": "linear", "bounds_error": False, "fill_value": "extrapolate"}

        ze = self.calc_ze(zn)
        dati = np.zeros_like(z)
        if zn.ndim == 1:
            dati = interp1d(ze, data, **opt)(z)
        elif zn.ndim == 2:
            for j in range(zn.shape[0]):
                dati[j, :] = interp1d(ze[j, :], data[j, :], **opt)(z[j, :])
        return dati

    @property
    def _idx_f(self):
        nnodes_half = int(len(self.element_table[0]) / 2)
        n_vfaces = self.n_elements + 1
        idx_f = np.zeros((n_vfaces, nnodes_half), dtype=int)
        idx_e = self._idx_e
        idx_f[: self.n_elements, :] = idx_e[:, :nnodes_half]
        idx_f[self.n_elements, :] = idx_e[-1, nnodes_half:]
        return idx_f

    @property
    def _idx_e(self):
        nnodes_per_elem = len(self.element_table[0])
        idx_e = np.zeros((self.n_elements, nnodes_per_elem), dtype=int)

        for j in range(self.n_elements):
            nodes = self.element_table[j]
            for i in range(nnodes_per_elem):
                idx_e[j, i] = nodes[i]

        return idx_e

    def _calc_z_using_idx(self, zn, idx):
        if zn.ndim == 1:
            zf = zn[idx].mean(axis=1)
        elif zn.ndim == 2:
            n_steps = zn.shape[0]
            zf = np.zeros((n_steps, idx.shape[0]))
            for step in range(n_steps):
                zf[step, :] = zn[step, idx].mean(axis=1)

        return zf


class _GeometryFMVerticalProfilePlotter:
    def __init__(self, geometry: "GeometryFM2D") -> None:
        self.g = geometry

    def __call__(self, ax=None, figsize=None, **kwargs):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        x = self.g.node_coordinates[:, 0]
        y = self.g.node_coordinates[:, 1]
        ax.plot(x, y, **kwargs)
        return ax

    def mesh(self, title="Mesh", edge_color="0.5", **kwargs):

        v = np.full_like(self.g.element_coordinates[:, 0], np.nan)
        return _plot_vertical_profile(
            node_coordinates=self.g.node_coordinates,
            element_table=self.g.element_table,
            values=v,
            is_geo=self.g.is_geo,
            title=title,
            add_colorbar=False,
            edge_color=edge_color,
            cmin=0.0,
            cmax=1.0,
            **kwargs,
        )
