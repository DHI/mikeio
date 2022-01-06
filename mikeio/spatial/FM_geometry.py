import warnings
import numpy as np
from collections import namedtuple
from mikecore.DfsuFile import DfsuFileType
from .geometry import _Geometry, BoundingBox
from ..custom_exceptions import InvalidGeometry


class GeometryFM(_Geometry):
    def __init__(
        self,
        node_coordinates=None,
        element_table=None,
        codes=None,
        projection_string=None,
        dfsu_type=None,
    ) -> None:
        super().__init__()

        self._type = None  # None: mesh, 0: 2d-dfsu, 4:dfsu3dsigma, ...

        self._nc = None
        self._ec = None
        self._codes = None
        self._element_ids = None
        self._node_ids = None
        self._element_table = None
        self._n_axis = 1
        self._n_layers = None

        self._tree2d = None
        self._boundary_polylines = None
        self._geom2d = None

        if node_coordinates is not None:
            self._set_nodes(
                node_coordinates=node_coordinates,
                codes=codes,
                node_ids=None,
                projection_string=projection_string,
            )

        if element_table is not None:
            self._set_elements(
                element_table=element_table,
                element_ids=None,
                dfsu_type=dfsu_type,
            )

    def __repr__(self):
        out = []
        out.append("Flexible Mesh Geometry")
        if self.n_nodes:
            out.append(f"Number of nodes: {self.n_nodes}")
        if self.n_elements:
            out.append(f"Number of elements: {self.n_elements}")
        if self._n_layers:
            out.append(f"Number of layers: {self._n_layers}")
        if self._projstr:
            out.append(f"Projection: {self.projection_string}")
        return str.join("\n", out)

    # should projection string still be here?
    def _set_nodes(
        self, node_coordinates, codes=None, node_ids=None, projection_string=None
    ):
        self._nc = np.asarray(node_coordinates)
        if codes is None:
            codes = np.zeros(len(node_coordinates), dtype=int)
        self._codes = np.asarray(codes)
        if node_ids is None:
            node_ids = np.arange(len(codes))
        self._node_ids = np.asarray(node_ids)
        self._projstr = "LONG/LAT" if projection_string is None else projection_string

    def _set_elements(self, element_table, element_ids=None, dfsu_type=None):
        self._element_table = element_table
        if element_ids is None:
            element_ids = np.arange(len(element_table))
        self._element_ids = np.asarray(element_ids)

        if dfsu_type is None:
            # guess type
            if self.max_nodes_per_element < 5:
                dfsu_type = DfsuFileType.Dfsu2D
            else:
                dfsu_type = DfsuFileType.Dfsu3DSigma
        self._type = dfsu_type

    def _reindex(self):
        new_node_ids = np.arange(self.n_nodes)
        new_element_ids = np.arange(self.n_elements)
        node_dict = dict(zip(self._node_ids, new_node_ids))
        for j in range(self.n_elements):
            elem_nodes = self._element_table[j]
            new_elem_nodes = []
            for idx in elem_nodes:
                new_elem_nodes.append(node_dict[idx])
            self._element_table[j] = new_elem_nodes

        self._node_ids = new_node_ids
        self._element_ids = new_element_ids

    @property
    def node_coordinates(self):
        """Coordinates (x,y,z) of all nodes"""
        return self._nc

    @property
    def n_nodes(self):
        """Number of nodes"""
        return len(self._node_ids)

    @property
    def n_elements(self):
        """Number of elements"""
        return len(self._element_ids)

    @property
    def element_ids(self):
        return self._element_ids

    @property
    def element_table(self):
        """Element to node connectivity"""
        if (self._element_table is None) and (self._element_table_mikecore is not None):
            self._element_table = self._get_element_table_from_mikecore()
        return self._element_table

    # cache this?
    @property
    def max_nodes_per_element(self):
        """The maximum number of nodes for an element"""
        maxnodes = 0
        for local_nodes in self.element_table:
            n = len(local_nodes)
            if n > maxnodes:
                maxnodes = n
        return maxnodes

    @property
    def type_name(self):
        """Type name, e.g. Mesh, Dfsu2D"""
        return self._type.name if self._type else "Mesh"

    @property
    def is_2d(self):
        """Type is either mesh or Dfsu2D (2 horizontal dimensions)"""
        return self._type in (
            DfsuFileType.Dfsu2D,
            DfsuFileType.DfsuSpectral2D,
            None,
        )

    @property
    def is_layered(self):
        """Type is layered dfsu (3d, vertical profile or vertical column)"""
        return self._type in (
            DfsuFileType.DfsuVerticalColumn,
            DfsuFileType.DfsuVerticalProfileSigma,
            DfsuFileType.DfsuVerticalProfileSigmaZ,
            DfsuFileType.Dfsu3DSigma,
            DfsuFileType.Dfsu3DSigmaZ,
        )

    @property
    def is_spectral(self):
        """Type is spectral dfsu (point, line or area spectrum)"""
        return self._type in (
            DfsuFileType.DfsuSpectral0D,
            DfsuFileType.DfsuSpectral1D,
            DfsuFileType.DfsuSpectral2D,
        )

    @property
    def is_tri_only(self):
        """Does the mesh consist of triangles only?"""
        return self.max_nodes_per_element == 3 or self.max_nodes_per_element == 6

    @property
    def element_coordinates(self):
        """Center coordinates of each element"""
        if self._ec is None:
            self._ec = self._calc_element_coordinates()
        return self._ec

    def _calc_element_coordinates(self, elements=None, zn=None):
        node_coordinates = self._nc

        element_table = self.element_table
        if elements is not None:
            element_table = element_table[elements]
        if zn is not None:
            node_coordinates = node_coordinates.copy()
            if len(zn) == len(node_coordinates[:, 2]):
                node_coordinates[:, 2] = zn
            else:
                # assume that user wants to find coords on a subset of points
                idx = np.unique(np.hstack(element_table))
                node_coordinates[idx, 2] = zn

        n_elements = len(element_table)
        ec = np.empty([n_elements, 3])

        # pre-allocate for speed
        maxnodes = 4 if self.is_2d else 8
        idx = np.zeros(maxnodes, dtype=int)
        xcoords = np.zeros([maxnodes, n_elements])
        ycoords = np.zeros([maxnodes, n_elements])
        zcoords = np.zeros([maxnodes, n_elements])
        nnodes_per_elem = np.zeros(n_elements)

        for j in range(n_elements):
            nodes = element_table[j]
            nnodes = len(nodes)
            nnodes_per_elem[j] = nnodes
            for i in range(nnodes):
                idx[i] = nodes[i]  # - 1

            xcoords[:nnodes, j] = node_coordinates[idx[:nnodes], 0]
            ycoords[:nnodes, j] = node_coordinates[idx[:nnodes], 1]
            zcoords[:nnodes, j] = node_coordinates[idx[:nnodes], 2]

        ec[:, 0] = np.sum(xcoords, axis=0) / nnodes_per_elem
        ec[:, 1] = np.sum(ycoords, axis=0) / nnodes_per_elem
        ec[:, 2] = np.sum(zcoords, axis=0) / nnodes_per_elem

        return ec

    @property
    def codes(self):
        """Node codes of all nodes (0=water, 1=land, 2...=open boundaries)"""
        return self._codes

    @property
    def boundary_polylines(self):
        """Lists of closed polylines defining domain outline"""
        if self._boundary_polylines is None:
            self._boundary_polylines = self._get_boundary_polylines()
        return self._boundary_polylines

    def contains(self, points):
        """test if a list of points are contained by mesh

        Parameters
        ----------
        points : array-like n-by-2
            x,y-coordinates of n points to be tested

        Returns
        -------
        bool array
            True for points inside, False otherwise
        """
        import matplotlib.path as mp

        points = np.atleast_2d(points)

        exterior = self.boundary_polylines.exteriors[0]
        cnts = mp.Path(exterior.xy).contains_points(points)

        if self.boundary_polylines.n_exteriors > 1:
            # in case of several dis-joint outer domains
            for exterior in self.boundary_polylines.exteriors[1:]:
                in_domain = mp.Path(exterior.xy).contains_points(points)
                cnts = np.logical_or(cnts, in_domain)

        # subtract any holes
        for interior in self.boundary_polylines.interiors:
            in_hole = mp.Path(interior.xy).contains_points(points)
            cnts = np.logical_and(cnts, ~in_hole)

        return cnts

    def _get_boundary_polylines_uncategorized(self):
        """Construct closed polylines for all boundary faces"""
        boundary_faces = self._get_boundary_faces()
        face_remains = boundary_faces.copy()
        polylines = []
        while face_remains.shape[0] > 1:
            n0 = face_remains[:, 0]
            n1 = face_remains[:, 1]
            polyline = [n0[0], n1[0]]
            index_to_delete = [0]
            count = 0
            end_points = face_remains[0, 1]
            while True:
                next_point_index = np.where(n0 == end_points)
                if next_point_index[0].size != 0:
                    polyline.append(face_remains[next_point_index[0][0], 1])
                    index_to_delete.append(next_point_index[0][0])
                    end_points = polyline[-1]
                count += 1
                if count > face_remains.shape[0] or polyline[0] == end_points:
                    break

            face_remains = np.delete(face_remains, index_to_delete, axis=0)
            polylines.append(polyline)
        return polylines

    def _get_boundary_polylines(self):
        """Get boundary polylines and categorize as inner or outer by
        assessing the signed area
        """
        polylines = self._get_boundary_polylines_uncategorized()

        poly_lines_int = []
        poly_lines_ext = []
        Polyline = namedtuple("Polyline", ["n_nodes", "nodes", "xy", "area"])

        for polyline in polylines:
            xy = self._geometry2d.node_coordinates[polyline, :2]
            area = (
                np.dot(xy[:, 1], np.roll(xy[:, 0], 1))
                - np.dot(xy[:, 0], np.roll(xy[:, 1], 1))
            ) * 0.5
            poly_line = np.asarray(polyline)
            xy = self._geometry2d.node_coordinates[poly_line, 0:2]
            poly = Polyline(len(polyline), poly_line, xy, area)
            if area > 0:
                poly_lines_ext.append(poly)
            else:
                poly_lines_int.append(poly)

        BoundaryPolylines = namedtuple(
            "BoundaryPolylines",
            ["n_exteriors", "exteriors", "n_interiors", "interiors"],
        )
        n_ext = len(poly_lines_ext)
        n_int = len(poly_lines_int)
        return BoundaryPolylines(n_ext, poly_lines_ext, n_int, poly_lines_int)

    def _get_boundary_faces(self):
        """Construct list of faces"""
        element_table = self._geometry2d.element_table

        all_faces = []
        for el in element_table:
            ele = [*el, el[0]]
            for j in range(len(el)):
                all_faces.append(ele[j : j + 2])

        all_faces = np.asarray(all_faces)

        all_faces_sorted = np.sort(all_faces, axis=1)
        _, uf_id, face_counts = np.unique(
            all_faces_sorted, axis=0, return_index=True, return_counts=True
        )

        # boundary faces are those appearing only once
        bnd_face_id = face_counts == 1
        return all_faces[uf_id[bnd_face_id]]

    def elements_to_geometry(self, elements, node_layers="all"):
        """export elements to new flexible file geometry

        Parameters
        ----------
        elements : list(int)
            list of element ids
        node_layers : str, optional
            for 3d files either 'top', 'bottom' layer nodes
            or 'all' can be selected, by default 'all'

        Returns
        -------
        UnstructuredGeometry
            which can be used for further extraction or saved to file
        """
        elements = np.sort(elements)  # make sure elements are sorted!

        # extract information for selected elements
        node_ids, elem_tbl = self._get_nodes_and_table_for_elements(
            elements, node_layers=node_layers
        )
        node_coords = self.node_coordinates[node_ids]
        codes = self.codes[node_ids]

        # create new geometry
        new_type = self._type
        if self.is_layered:
            layers_used = self.layer_ids[elements]
            unique_layer_ids = np.unique(layers_used)
            n_layers = len(unique_layer_ids)
            if (
                self._type == DfsuFileType.Dfsu3DSigma
                or self._type == DfsuFileType.Dfsu3DSigmaZ
            ) and n_layers == 1:
                new_type = DfsuFileType.Dfsu2D

        if self.is_layered and (new_type != DfsuFileType.Dfsu2D):
            geom = GeometryFMLayered()
        else:
            geom = GeometryFM()

        geom._set_nodes(
            node_coords,
            codes=codes,
            node_ids=node_ids,
            projection_string=self.projection_string,
        )
        geom._set_elements(elem_tbl, self.element_ids[elements])
        geom._reindex()

        geom._type = self._type  #
        if self.is_layered:
            if new_type == DfsuFileType.Dfsu2D:
                # If source is 3d, but output only has 1 layer
                # then change type to 2d
                geom._type = DfsuFileType.Dfsu2D
                geom._n_layers = None
                if node_layers == "all":
                    warnings.warn(
                        "Warning: Only 1 layer in new geometry (hence 2d), but you have kept both top and bottom nodes! Hint: use node_layers='top' or 'bottom'"
                    )
            else:
                geom._type = self._type
                geom._n_layers = n_layers
                lowest_sigma = self.n_layers - self.n_sigma_layers + 1
                geom._n_sigma = sum(unique_layer_ids >= lowest_sigma)

                # If source is sigma-z but output only has sigma layers
                # then change type accordingly
                if (
                    self._type == DfsuFileType.DfsuVerticalProfileSigmaZ
                    or self._type == DfsuFileType.Dfsu3DSigmaZ
                ) and n_layers == geom._n_sigma:
                    # TODO fix this
                    geom._type = DfsuFileType.Dfsu3DSigma

                geom._top_elems = geom._get_top_elements_from_coordinates()

        return geom

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
        nodes = []
        elem_tbl = []
        if (node_layers is None) or (node_layers == "all") or self.is_2d:
            for j in elements:
                elem_nodes = self.element_table[j]
                elem_tbl.append(elem_nodes)
                for node in elem_nodes:
                    nodes.append(node)
        else:
            # 3D file
            if (node_layers != "bottom") and (node_layers != "top"):
                raise Exception("node_layers must be either all, bottom or top")
            for j in elements:
                elem_nodes = self.element_table[j]
                nn = len(elem_nodes)
                halfn = int(nn / 2)
                if node_layers == "bottom":
                    elem_nodes = elem_nodes[:halfn]
                if node_layers == "top":
                    elem_nodes = elem_nodes[halfn:]
                elem_tbl.append(elem_nodes)
                for node in elem_nodes:
                    nodes.append(node)

        return np.unique(nodes), elem_tbl

    # 3D dfsu stuff
    @property
    def _geometry2d(self):
        """The 2d geometry for a 3d object"""
        if self._n_layers is None:
            return self
        if self._geom2d is None:
            self._geom2d = self.to_2d_geometry()
        return self._geom2d

    def _to_polygons(self, geometry=None):
        """generate matplotlib polygons from element table for plotting

        Returns
        -------
        list(matplotlib.patches.Polygon)
            list of polygons for plotting
        """
        if geometry is None:
            geometry = self
        from matplotlib.patches import Polygon

        polygons = []

        for j in range(geometry.n_elements):
            nodes = geometry.element_table[j]
            pcoords = np.empty([len(nodes), 2])
            for i in range(len(nodes)):
                nidx = nodes[i]
                pcoords[i, :] = geometry.node_coordinates[nidx, 0:2]

            polygon = Polygon(pcoords, True)
            polygons.append(polygon)
        return polygons

    def plot_mesh(self, figsize=None, ax=None):
        import matplotlib.pyplot as plt
        from matplotlib.collections import PatchCollection

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        ax.set_aspect(self._plot_aspect())

        patches = self._geometry2d._to_polygons()
        fig_obj = PatchCollection(
            patches, edgecolor="0.6", facecolor="none", linewidths=0.3
        )
        ax.add_collection(fig_obj)
        self.plot_outline(ax=ax)
        return ax

    def plot_outline(self, figsize=None, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        ax.set_aspect(self._plot_aspect())

        linwid = 1.2
        out_col = "0.4"
        for exterior in self.boundary_polylines.exteriors:
            ax.plot(*exterior.xy.T, color=out_col, linewidth=linwid)
        for interior in self.boundary_polylines.interiors:
            ax.plot(*interior.xy.T, color=out_col, linewidth=linwid)
        return ax

    def plot_boundary_nodes(self, boundary_names=None, figsize=None, ax=None):
        """
        Plot mesh boundary nodes and their codes
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect(self._plot_aspect())

        nc = self.node_coordinates
        c = self.codes
        valid_codes = list(set(self.codes))
        boundary_codes = [code for code in valid_codes if code > 0]

        if boundary_names is not None:
            if len(boundary_codes) != len(boundary_names):
                raise Exception(
                    f"Number of boundary names ({len(boundary_names)}) inconsistent with number of boundaries ({len(self.boundary_codes)})"
                )
            user_defined_labels = dict(zip(boundary_codes, boundary_names))

        for code in boundary_codes:
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

    def _plot_aspect(self):
        if self.is_geo:
            mean_lat = np.mean(self.node_coordinates[:, 1])
            return 1.0 / np.cos(np.pi * mean_lat / 180)
        else:
            return "equal"


# class GeometryFMHorizontal(GeometryFM):
#     pass


class GeometryFMLayered(GeometryFM):
    def __init__(
        self,
        node_coordinates=None,
        element_table=None,
        codes=None,
        projection_string=None,
        dfsu_type=None,
    ) -> None:
        super().__init__(
            node_coordinates=node_coordinates,
            element_table=element_table,
            codes=codes,
            projection_string=projection_string,
            dfsu_type=dfsu_type,
        )
        self._top_elems = None
        self._n_layers_column = None
        self._bot_elems = None
        self._n_layers = None
        self._n_sigma = None

        self._geom2d = None
        self._e2_e3_table = None
        self._2d_ids = None
        self._layer_ids = None

    @property
    def geometry2d(self):
        """The 2d geometry for a 3d object"""
        return self._geometry2d()

    def to_2d_geometry(self):
        """extract 2d geometry from 3d geometry

        Returns
        -------
        UnstructuredGeometry
            2d geometry (bottom nodes)
        """
        if self.is_2d:
            return self

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
        geom = GeometryFM()
        geom._set_nodes(
            node_coords,
            codes=codes,
            node_ids=node_ids,
            projection_string=self.projection_string,
        )
        geom._set_elements(elem_tbl, self.element_ids[elem_ids])

        # TODO how to handle Mesh filetype
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
    def e2_e3_table(self):
        """The 2d-to-3d element connectivity table for a 3d object"""
        if self._n_layers is None:
            print("Object has no layers: cannot return e2_e3_table")
            return None
        if self._e2_e3_table is None:
            res = self._get_2d_to_3d_association()
            self._e2_e3_table = res[0]
            self._2d_ids = res[1]
            self._layer_ids = res[2]
        return self._e2_e3_table

    @property
    def elem2d_ids(self):
        """The associated 2d element id for each 3d element"""
        if self._n_layers is None:
            raise InvalidGeometry("Object has no layers: cannot return elem2d_ids")
            # or return self._2d_ids ??

        if self._2d_ids is None:
            res = self._get_2d_to_3d_association()
            self._e2_e3_table = res[0]
            self._2d_ids = res[1]
            self._layer_ids = res[2]
        return self._2d_ids

    @property
    def layer_ids(self):
        """The layer number (0=bottom, 1, 2, ...) for each 3d element"""
        if self._n_layers is None:
            raise InvalidGeometry("Object has no layers: cannot return layer_ids")
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
        if self._n_layers is None:
            return None
        return self._n_layers - self._n_sigma

    @property
    def top_elements(self):
        """List of 3d element ids of surface layer"""
        if self._n_layers is None:
            print("Object has no layers: cannot find top_elements")
            return None
        elif (self._top_elems is None) and (self._source is not None):
            # note: if subset of elements is selected then this cannot be done!
            self._top_elems = np.array(
                DfsuUtil.FindTopLayerElements(self._source.ElementTable)
            )
        return self._top_elems

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

    @property
    def bottom_elements(self):
        """List of 3d element ids of bottom layer"""
        if self._n_layers is None:
            print("Object has no layers: cannot find bottom_elements")
            return None
        elif self._bot_elems is None:
            self._bot_elems = self.top_elements - self.n_layers_per_column + 1
        return self._bot_elems

    def get_layer_elements(self, layer):
        """3d element ids for one (or more) specific layer(s)

        Parameters
        ----------
        layer : int or list(int)
            layer between 0 (bottom) and n_layers-1 (top)
            (can also be negative counting from -1 at the top layer)

        Returns
        -------
        np.array(int)
            element ids
        """
        if not np.isscalar(layer):
            elem_ids = []
            for nn in layer:
                elem_ids.append(self.get_layer_elements(nn))
            elem_ids = np.concatenate(elem_ids, axis=0)
            return np.sort(elem_ids)

        n_lay = self.n_layers
        if n_lay is None:
            raise InvalidGeometry("Object has no layers: cannot get_layer_elements")

        if layer < (-n_lay) or layer >= n_lay:
            raise Exception(
                f"Layer {layer} not allowed; must be between -{n_lay} and {n_lay-1}"
            )

        if layer < 0:
            layer = layer + n_lay

        return self.element_ids[self.layer_ids == layer]

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
            col = list(range(botid[j], topid[j] + 1))

            e2_to_e3.append(col)
            for jj in col:
                index2d.append(j)

            n_local_layers = len(col)
            local_layers = global_layer_ids[-n_local_layers:]
            for ll in local_layers:
                layerid.append(ll)

        e2_to_e3 = np.array(e2_to_e3, dtype=object)
        index2d = np.array(index2d)
        layerid = np.array(layerid)
        return e2_to_e3, index2d, layerid

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

    # TODO: add methods for extracting layers etc


# class GeometryFMSpectral(GeometryFM):
#     # TODO: add specialized classes: FrequencySpectrum, DirectionalSpectrum
#     def __init__(
#         self,
#         frequencies=None,
#         directions=None,
#         node_coordinates=None,
#         element_table=None,
#         codes=None,
#         projection_string=None,
#         dfsu_type=None,
#     ) -> None:
#         super().__init__(
#             node_coordinates=node_coordinates,
#             element_table=element_table,
#             codes=codes,
#             projection_string=projection_string,
#             dfsu_type=dfsu_type,
#         )
#         self._frequencies = frequencies
#         self._directions = directions
#         self._n_axis = 0 if (self.n_elements == 0) else 1
#         self._n_axis = (
#             self._n_axis + int(self.n_frequencies > 0) + int(self.n_directions > 0)
#         )

#     @property
#     def n_frequencies(self):
#         """Number of frequencies"""
#         return 0 if self.frequencies is None else len(self.frequencies)

#     @property
#     def frequencies(self):
#         """Frequency axis"""
#         return self._frequencies

#     @property
#     def n_directions(self):
#         """Number of directions"""
#         return 0 if self.directions is None else len(self.directions)

#     @property
#     def directions(self):
#         """Directional axis"""
#         return self._directions
