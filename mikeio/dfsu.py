import os
from collections import namedtuple
import pandas as pd
import pathlib
import warnings
import numpy as np
from datetime import datetime, timedelta
from functools import wraps

from scipy.spatial import cKDTree
import tempfile
from tqdm import trange
import typing

from mikecore.eum import eumUnit, eumQuantity
from mikecore.DfsFactory import DfsFactory
from mikecore.DfsuBuilder import DfsuBuilder
from mikecore.DfsuFile import DfsuFile, DfsuFileType

from mikecore.MeshFile import MeshFile
from mikecore.MeshBuilder import MeshBuilder

from .dfsutil import _get_item_info, _valid_item_numbers, _valid_timesteps
from .dataset import Dataset, DataArray
from .dfs0 import Dfs0
from .dfs2 import Dfs2
from .eum import ItemInfo, EUMType, EUMUnit
from .spatial.FM_geometry import GeometryFM, GeometryFMLayered

from .spatial import Grid2D
from .custom_exceptions import InvalidGeometry
from .base import EquidistantTimeSeries
from .interpolation import get_idw_interpolant, interp2d


class Dfsu:
    def __new__(self, filename, *args, **kwargs):
        filename = str(filename)
        type = self._get_DfsuFileType(filename)

        if self._type_is_spectral(type):
            return DfsuSpectral(filename, *args, **kwargs)
        elif self._type_is_2d_horizontal(type):
            return Dfsu2DH(filename, *args, **kwargs)
        elif self._type_is_2d_vertical(type):
            return Dfsu2DV(filename, *args, **kwargs)
        elif self._type_is_3d(type):
            return Dfsu3D(filename, *args, **kwargs)
        else:
            raise ValueError(f"Type {type} is unsupported!")

    @staticmethod
    def _get_DfsuFileType(filename: str):
        ext = os.path.splitext(filename)[-1]
        if "dfs" in ext:
            dfs = DfsuFile.Open(filename)
            type = DfsuFileType(dfs.DfsuFileType)
            dfs.Close()
        elif "mesh" in ext:
            type = None
        else:
            raise ValueError(f"{ext} is an unsupported extension")
        return type

    @staticmethod
    def _type_is_2d_horizontal(type):
        return type in (
            DfsuFileType.Dfsu2D,
            DfsuFileType.DfsuSpectral2D,
            None,
        )

    @staticmethod
    def _type_is_2d_vertical(type):
        return type in (
            DfsuFileType.DfsuVerticalProfileSigma,
            DfsuFileType.DfsuVerticalProfileSigmaZ,
        )

    @staticmethod
    def _type_is_3d(type):
        return type in (
            DfsuFileType.Dfsu3DSigma,
            DfsuFileType.Dfsu3DSigmaZ,
        )

    @staticmethod
    def _type_is_spectral(type):
        """Type is spectral dfsu (point, line or area spectrum)"""
        return type in (
            DfsuFileType.DfsuSpectral0D,
            DfsuFileType.DfsuSpectral1D,
            DfsuFileType.DfsuSpectral2D,
        )


class _UnstructuredGeometry:
    def __init__(self) -> None:
        self._type = None  # -1: mesh, 0: 2d-dfsu, 4:dfsu3dsigma, ...
        self._geometry = None
        self._geom2d = None
        self._shapely_domain_obj = None

    @property
    def type_name(self):
        """Type name, e.g. Mesh, Dfsu2D"""
        return self._type.name if self._type else "Mesh"

    @property
    def geometry(self):
        return self._geometry

    @property
    def _geometry2d(self):
        """The 2d geometry for a 3d object or geometry for a 2d object"""
        if self._geometry.is_2d:
            return self.geometry
        if self._geom2d is None:
            self._geom2d = self.geometry.to_2d_geometry()
        return self._geom2d

    @property
    def n_nodes(self):
        """Number of nodes"""
        return self.geometry.n_nodes

    @property
    def node_coordinates(self):
        """Coordinates (x,y,z) of all nodes"""
        return self.geometry.node_coordinates

    @property
    def node_ids(self):
        return self.geometry.node_ids

    @property
    def n_elements(self):
        """Number of elements"""
        return self.geometry.n_elements

    @property
    def element_ids(self):
        return self.geometry.element_ids

    @property
    def codes(self):
        """Node codes of all nodes (0=water, 1=land, 2...=open boundaries)"""
        return self.geometry.codes

    @codes.setter
    def codes(self, v):
        if len(v) != self.n_nodes:
            raise ValueError(f"codes must have length of nodes ({self.n_nodes})")
        self._geometry._codes = np.array(v, dtype=np.int32)

    @property
    def valid_codes(self):
        """Unique list of node codes"""
        return list(set(self.codes))

    @property
    def boundary_codes(self):
        """Unique list of boundary codes"""
        return [code for code in self.valid_codes if code > 0]

    @property
    def projection_string(self):
        """The projection string"""
        return self.geometry.projection_string

    @property
    def is_geo(self):
        """Are coordinates geographical (LONG/LAT)?"""
        return self.geometry.projection_string == "LONG/LAT"

    @property
    def is_local_coordinates(self):
        """Are coordinates relative (NON-UTM)?"""
        return self.geometry.projection_string == "NON-UTM"

    @property
    def element_table(self):
        """Element to node connectivity"""
        return self.geometry.element_table

    @property
    def max_nodes_per_element(self):
        """The maximum number of nodes for an element"""
        return self.geometry.max_nodes_per_element

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
        return self.geometry.is_tri_only

    @property
    def boundary_polylines(self):
        """Lists of closed polylines defining domain outline"""
        return self.geometry.boundary_polylines

    def get_node_coords(self, code=None):
        """Get the coordinates of each node.

        Parameters
        ----------
        code: int
            Get only nodes with specific code, e.g. land == 1

        Returns
        -------
        np.array
            x,y,z of each node
        """
        nc = self.node_coordinates
        if code is not None:
            if code not in self.valid_codes:
                print(
                    f"Selected code: {code} is not valid. Valid codes: {self.valid_codes}"
                )
                raise Exception
            return nc[self.codes == code]
        return nc

    @wraps(GeometryFM.elements_to_geometry)
    def elements_to_geometry(self, elements, node_layers="all"):
        return self.geometry.elements_to_geometry(elements, node_layers)

    @staticmethod
    def _offset_element_table_by(element_table, offset):
        offset = int(offset)
        new_elem_table = element_table.copy()
        for j in range(len(element_table)):
            new_elem_table[j] = np.array(element_table[j]) + offset
        return new_elem_table

    @staticmethod
    def _get_element_table_from_mikecore(element_table):
        return _UnstructuredGeometry._offset_element_table_by(element_table, -1)

    @staticmethod
    def _element_table_to_mikecore(element_table):
        return _UnstructuredGeometry._offset_element_table_by(element_table, 1)

    @property
    def element_coordinates(self):
        """Center coordinates of each element"""
        return self.geometry.element_coordinates

    @wraps(GeometryFMLayered.calc_element_coordinates)
    def calc_element_coordinates(self, elements=None, zn=None):
        return self.geometry.calc_element_coordinates(elements, zn)

    @wraps(GeometryFM.contains)
    def contains(self, points):
        return self.geometry.contains(points)

    def get_overset_grid(self, dx=None, dy=None, shape=None, buffer=None):
        """get a 2d grid that covers the domain by specifying spacing or shape

        Parameters
        ----------
        dx : float or (float, float), optional
            grid resolution in x-direction (or in x- and y-direction)
        dy : float, optional
            grid resolution in y-direction
        shape : (int, int), optional
            tuple with nx and ny describing number of points in each direction
            one of them can be None, in which case the value will be inferred
        buffer : float, optional
            positive to make the area larger, default=0
            can be set to a small negative value to avoid NaN
            values all around the domain.

        Returns
        -------
        <mikeio.Grid2D>
            2d grid
        """
        nc = self._geometry2d.node_coordinates
        bbox = Grid2D.xy_to_bbox(nc, buffer=buffer)
        return Grid2D(bbox=bbox, dx=dx, dy=dy, shape=shape)

    @wraps(GeometryFM.get_2d_interpolant)
    def get_2d_interpolant(
        self, xy, n_nearest: int = 1, extrapolate=False, p=2, radius=None
    ):
        return self.geometry.get_2d_interpolant(xy, n_nearest, extrapolate, p, radius)

    @wraps(GeometryFM.interp2d)
    def interp2d(self, data, elem_ids, weights=None, shape=None):
        return self.geometry.interp2d(data, elem_ids, weights, shape)

    @wraps(GeometryFM.find_nearest_elements)
    def find_nearest_elements(
        self, x, y=None, z=None, layer=None, n_nearest=1, return_distances=False
    ):
        return self.geometry.find_nearest_elements(
            x, y, z, layer, n_nearest, return_distances
        )

    @wraps(GeometryFM.get_element_area)
    def get_element_area(self):
        return self.geometry.get_element_area()

    @wraps(GeometryFM._to_polygons)
    def _to_polygons(self, geometry=None):
        return geometry._to_polygons(geometry)

    @wraps(GeometryFM.to_shapely)
    def to_shapely(self):
        return self.geometry.to_shapely()

    @wraps(GeometryFM.get_node_centered_data)
    def get_node_centered_data(self, data, extrapolate=True):
        return self.geometry.get_node_centered_data(data, extrapolate)

    def plot(
        self,
        z=None,
        elements=None,
        plot_type="patch",
        title=None,
        label=None,
        cmap=None,
        vmin=None,
        vmax=None,
        levels=None,
        n_refinements=0,
        show_mesh=True,
        show_outline=True,
        figsize=None,
        ax=None,
        add_colorbar=True,
    ):
        """
        Plot unstructured data and/or mesh, mesh outline

        Parameters
        ----------
        z: np.array or a Dataset with a single item, optional
            value for each element to plot, default bathymetry
        elements: list(int), optional
            list of element ids to be plotted
        plot_type: str, optional
            type of plot: 'patch' (default), 'mesh_only', 'shaded',
            'contour', 'contourf' or 'outline_only'
        title: str, optional
            axes title
        label: str, optional
            colorbar label (or title if contour plot)
        cmap: matplotlib.cm.cmap, optional
            colormap, default viridis
        vmin: real, optional
            lower bound of values to be shown on plot, default:None
        vmax: real, optional
            upper bound of values to be shown on plot, default:None
        levels: int, list(float), optional
            for contour plots: how many levels, default:10
            or a list of discrete levels e.g. [3.0, 4.5, 6.0]
        show_mesh: bool, optional
            should the mesh be shown on the plot? default=True
        show_outline: bool, optional
            should domain outline be shown on the plot? default=True
        n_refinements: int, optional
            for 'shaded' and 'contour' plots (and if show_mesh=False)
            do this number of mesh refinements for smoother plotting
        figsize: (float, float), optional
            specify size of figure
        ax: matplotlib.axes, optional
            Adding to existing axis, instead of creating new fig
        add_colorbar: bool
            Add colorbar to plot, default True

        Returns
        -------
        <matplotlib.axes>

        Examples
        --------
        >>> dfs = Dfsu("HD2D.dfsu")
        >>> dfs.plot() # bathymetry
        >>> ds = dfs.read(items="Surface elevation", time_steps=0)
        >>> ds.shape
        (1, 884)
        >>> ds.n_items
        1
        >>> dfs.plot(z=ds) # plot surface elevation
        """

        import matplotlib.cm as cm
        import matplotlib.colors as mplc
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        from matplotlib.collections import PatchCollection
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        mesh_col = "0.95"
        mesh_col_dark = "0.6"

        if plot_type is None:
            plot_type = "outline_only"

        plot_data = True
        if plot_type == "mesh_only" or plot_type == "outline_only":
            plot_data = False

        if cmap is None:
            cmap = cm.viridis

        if elements is None:
            if self.is_2d:
                geometry = self.geometry
            else:
                geometry = self._geometry2d
        else:
            # spatial subset
            if self.is_2d:
                geometry = self.geometry.elements_to_geometry(elements)
            else:
                geometry = self.geometry.elements_to_geometry(
                    elements, node_layers="bottom"
                )

        nc = geometry.node_coordinates
        ec = geometry.element_coordinates
        ne = ec.shape[0]

        is_bathy = False
        if z is None:
            is_bathy = True
            if plot_data:
                z = ec[:, 2]
                if label is None:
                    label = "Bathymetry (m)"
        else:
            if isinstance(z, DataArray):
                z = z.to_numpy().copy()
            if isinstance(z, Dataset) and len(z) == 1:  # if single-item Dataset
                z = z[0].to_numpy().copy()
            if len(z) != ne:
                z = np.squeeze(z).copy()  # handles single time step
                if len(z) != ne:
                    raise Exception(
                        f"Length of z ({len(z)}) does not match geometry ({ne})"
                    )
            if label is None:
                label = ""
            if not plot_data:
                print(f"Cannot plot data in {plot_type} plot!")

        if plot_data and vmin is None:
            vmin = np.nanmin(z)
        if plot_data and vmax is None:
            vmax = np.nanmax(z)

        # set levels
        cmap_norm = None
        cmap_ScMappable = None
        if ("only" not in plot_type) and (levels is not None):
            if np.isscalar(levels):
                n_levels = levels
                levels = np.linspace(vmin, vmax, n_levels)
            else:
                n_levels = len(levels)
                vmin = min(levels)
                vmax = max(levels)

            levels = np.array(levels)
            levels_equidistant = all(np.diff(levels) == np.diff(levels)[0])
            # if not levels_equidistant:  # (isinstance(cmap, mplc.ListedColormap)) or (
            if (not isinstance(cmap, str)) and (not levels_equidistant):
                # print("ScalarMappable")
                # cmap = mplc.Colormap(cmap)
                cmap_norm = mplc.BoundaryNorm(levels, cmap.N)
                cmap_ScMappable = cm.ScalarMappable(cmap=cmap, norm=cmap_norm)
        if ("contour" in plot_type) and (levels is None):
            levels = 10
            n_levels = 10

        cbar_extend = self._cbar_extend(z, vmin, vmax)

        # plot in existing or new axes?
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # set aspect ratio
        if geometry.is_geo:
            mean_lat = np.mean(nc[:, 1])
            ax.set_aspect(1.0 / np.cos(np.pi * mean_lat / 180))
        else:
            ax.set_aspect("equal")

        # set plot limits
        xmin, xmax = nc[:, 0].min(), nc[:, 0].max()
        ymin, ymax = nc[:, 1].min(), nc[:, 1].max()

        # scale height of colorbar
        cbar_frac = 0.046 * nc[:, 1].ptp() / nc[:, 0].ptp()

        if plot_type == "outline_only":
            fig_obj = None

        elif plot_type == "mesh_only":
            if show_mesh == False:
                print("Not possible to use show_mesh=False on a mesh_only plot!")
            patches = geometry._to_polygons()
            fig_obj = PatchCollection(
                patches, edgecolor=mesh_col_dark, facecolor="none", linewidths=0.3
            )
            ax.add_collection(fig_obj)

        elif plot_type == "patch" or plot_type == "box":
            patches = geometry._to_polygons()
            # do plot as patches (like MZ "box contour")
            # with (constant) element center values
            if show_mesh:
                fig_obj = PatchCollection(
                    patches,
                    cmap=cmap,
                    norm=cmap_norm,
                    edgecolor=mesh_col,
                    linewidths=0.4,
                )
            else:
                fig_obj = PatchCollection(
                    patches,
                    cmap=cmap,
                    norm=cmap_norm,
                    edgecolor="face",
                    alpha=None,
                    linewidths=None,
                )

            fig_obj.set_array(z)
            fig_obj.set_clim(vmin, vmax)
            ax.add_collection(fig_obj)

            if add_colorbar:
                cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
                cmap_sm = cmap_ScMappable if cmap_ScMappable else fig_obj
                plt.colorbar(cmap_sm, label=label, cax=cax, extend=cbar_extend)

        else:
            # do node-based triangular plot
            import matplotlib.tri as tri

            mesh_linewidth = 0.0
            if show_mesh and geometry.is_tri_only:
                mesh_linewidth = 0.4
                if n_refinements > 0:
                    n_refinements = 0
                    print("Warning: mesh refinement is not possible if plot_mesh=True")

            elem_table, ec, z = geometry._create_tri_only_element_table(data=z)
            triang = tri.Triangulation(nc[:, 0], nc[:, 1], elem_table)

            zn = geometry.get_node_centered_data(z)

            if n_refinements > 0:
                # TODO: refinements doesn't seem to work for 3d files?
                refiner = tri.UniformTriRefiner(triang)
                triang, zn = refiner.refine_field(zn, subdiv=n_refinements)

            if plot_type == "shaded" or plot_type == "smooth":
                ax.triplot(triang, lw=mesh_linewidth, color=mesh_col)
                if cmap_norm is None:
                    fig_obj = ax.tripcolor(
                        triang,
                        zn,
                        edgecolors="face",
                        vmin=vmin,
                        vmax=vmax,
                        cmap=cmap,
                        linewidths=0.3,
                        shading="gouraud",
                    )
                else:
                    fig_obj = ax.tripcolor(
                        triang,
                        zn,
                        edgecolors="face",
                        cmap=cmap,
                        norm=cmap_norm,
                        linewidths=0.3,
                        shading="gouraud",
                    )

                if add_colorbar:
                    cax = make_axes_locatable(ax).append_axes(
                        "right", size="5%", pad=0.05
                    )
                    cmap_sm = cmap_ScMappable if cmap_ScMappable else fig_obj

                    plt.colorbar(
                        cmap_sm,
                        label=label,
                        cax=cax,
                        boundaries=levels,
                        extend=cbar_extend,
                    )

            elif plot_type == "contour" or plot_type == "contour_lines":
                ax.triplot(triang, lw=mesh_linewidth, color=mesh_col_dark)
                fig_obj = ax.tricontour(
                    triang,
                    zn,
                    levels=levels,
                    linewidths=[1.2],
                    cmap=cmap,
                    norm=cmap_norm,
                )
                ax.clabel(fig_obj, fmt="%1.2f", inline=1, fontsize=9)
                if len(label) > 0:
                    ax.set_title(label)

            elif plot_type == "contourf" or plot_type == "contour_filled":
                ax.triplot(triang, lw=mesh_linewidth, color=mesh_col)
                vbuf = 0.01 * (vmax - vmin) / n_levels
                # avoid white outside limits
                zn = np.clip(zn, vmin + vbuf, vmax - vbuf)
                fig_obj = ax.tricontourf(
                    triang,
                    zn,
                    levels=levels,
                    cmap=cmap,
                    norm=cmap_norm,
                    extend=cbar_extend,
                )

                # colorbar
                if add_colorbar:
                    cax = make_axes_locatable(ax).append_axes(
                        "right", size="5%", pad=0.05
                    )
                    if cmap_ScMappable is None:
                        plt.colorbar(fig_obj, label=label, cax=cax)
                    else:
                        try:
                            plt.colorbar(
                                cmap_ScMappable,
                                label=label,
                                cax=cax,
                                ticks=levels,
                            )
                        except:
                            warnings.warn("Cannot add colorbar")

            else:
                if (plot_type is not None) and plot_type != "outline_only":
                    raise Exception(f"plot_type {plot_type} unknown!")

            if show_mesh and (not geometry.is_tri_only):
                # if mesh is not tri only, we need to add it manually on top
                patches = geometry._to_polygons()
                mesh_linewidth = 0.4
                if plot_type == "contour":
                    mesh_col = mesh_col_dark
                p = PatchCollection(
                    patches,
                    edgecolor=mesh_col,
                    facecolor="none",
                    linewidths=mesh_linewidth,
                )
                ax.add_collection(p)

        if show_outline:
            linwid = 1.2
            out_col = "0.4"
            for exterior in self.boundary_polylines.exteriors:
                ax.plot(*exterior.xy.T, color=out_col, linewidth=linwid)
                xd, yd = exterior.xy[:, 0], exterior.xy[:, 1]
                xmin, xmax = min(xmin, np.min(xd)), max(xmax, np.max(xd))
                ymin, ymax = min(ymin, np.min(yd)), max(ymax, np.max(yd))

            for interior in self.boundary_polylines.interiors:
                ax.plot(*interior.xy.T, color=out_col, linewidth=linwid)

        # set plot limits
        xybuf = 6e-3 * (xmax - xmin)
        ax.set_xlim(xmin - xybuf, xmax + xybuf)
        ax.set_ylim(ymin - xybuf, ymax + xybuf)

        if title is not None:
            ax.set_title(title)

        return ax

    def _cbar_extend(self, calc_data, vmin, vmax):
        if calc_data is None:
            return "neither"
        extend_min = calc_data.min() < vmin if vmin is not None else False
        extend_max = calc_data.max() > vmax if vmax is not None else False
        if extend_min and extend_max:
            extend = "both"
        elif extend_min:
            extend = "min"
        elif extend_max:
            extend = "max"
        else:
            extend = "neither"
        return extend


class _UnstructuredFile(_UnstructuredGeometry):
    """
    _UnstructuredFile based on _UnstructuredGeometry and base class for Mesh and Dfsu
    has file handle, items and timesteps and reads file header
    """

    show_progress = False

    def __repr__(self):
        out = []
        if self._type is not None:
            out.append(self.type_name)
        if self._type is not DfsuFileType.DfsuSpectral0D:
            if self._type is not DfsuFileType.DfsuSpectral1D:
                out.append(f"Number of elements: {self.n_elements}")
            out.append(f"Number of nodes: {self.n_nodes}")
        if self.is_spectral:
            if self.n_frequencies > 0:
                out.append(f"Number of frequencies: {self.n_frequencies}")
            if self.n_directions > 0:
                out.append(f"Number of directions: {self.n_directions}")
        if self.geometry.projection_string:
            out.append(f"Projection: {self.projection_string}")
        if self.is_layered:
            out.append(f"Number of sigma layers: {self.n_sigma_layers}")
        if (
            self._type == DfsuFileType.DfsuVerticalProfileSigmaZ
            or self._type == DfsuFileType.Dfsu3DSigmaZ
        ):
            out.append(f"Max number of z layers: {self.n_layers - self.n_sigma_layers}")
        if self._n_items is not None:
            if self._n_items < 10:
                out.append("Items:")
                for i, item in enumerate(self.items):
                    out.append(f"  {i}:  {item}")
            else:
                out.append(f"Number of items: {self._n_items}")
        if self._n_timesteps is not None:
            if self._n_timesteps == 1:
                out.append(f"Time: time-invariant file (1 step) at {self._start_time}")
            else:
                out.append(
                    f"Time: {self._n_timesteps} steps with dt={self._timestep_in_seconds}s"
                )
                out.append(f"      {self._start_time} -- {self.end_time}")
        return str.join("\n", out)

    def __init__(self):
        super().__init__()
        self._filename = None
        self._source = None
        self._deletevalue = None

        self._n_timesteps = None
        self._start_time = None
        self._timestep_in_seconds = None

        self._n_items = None
        self._items = None
        self._dtype = np.float64

        self._geometry = None

    def _read_header(self, filename):
        if not os.path.isfile(filename):
            raise Exception(f"file {filename} does not exist!")

        _, ext = os.path.splitext(filename)

        if ext == ".mesh":
            self._read_mesh_header(filename)

        elif ext == ".dfsu":
            self._read_dfsu_header(filename)
        else:
            raise Exception(f"Filetype {ext} not supported (mesh,dfsu)")

    def _read_mesh_header(self, filename):
        """
        Read header of mesh file and set object properties
        """
        msh = MeshFile.ReadMesh(filename)
        self._source = msh
        # self._projstr = msh.ProjectionString
        self._type = None  # DfsuFileType.Mesh

        nc, codes, node_ids = self._get_nodes_from_source(msh)
        el_table, el_ids = self._get_elements_from_source(msh)

        # create geometry
        self._geometry = GeometryFM(
            node_coordinates=nc,
            element_table=el_table,
            codes=codes,
            projection_string=msh.ProjectionString,
            dfsu_type=self._type,
            element_ids=el_ids,
            node_ids=node_ids,
        )

    def _read_dfsu_header(self, filename):
        """
        Read header of dfsu file and set object properties
        """
        dfs = DfsuFile.Open(filename)
        self._source = dfs
        self._type = DfsuFileType(dfs.DfsuFileType)
        self._deletevalue = dfs.DeleteValueFloat

        if self.is_spectral:
            dir = dfs.Directions
            self.directions = None if dir is None else dir * (180 / np.pi)
            self.n_directions = dfs.NumberOfDirections
            self.frequencies = dfs.Frequencies
            self.n_frequencies = dfs.NumberOfFrequencies

        # geometry
        if self._type == DfsuFileType.DfsuSpectral0D:
            self._geometry = GeometryFM()  # EMPTY
        else:
            nc, codes, node_ids = self._get_nodes_from_source(dfs)
            el_table, el_ids = self._get_elements_from_source(dfs)

            if self.is_layered:
                geometry_cls = GeometryFMLayered
                self._geometry = geometry_cls(
                    node_coordinates=nc,
                    element_table=el_table,
                    codes=codes,
                    projection_string=dfs.Projection.WKTString,
                    dfsu_type=self._type,
                    element_ids=el_ids,
                    node_ids=node_ids,
                    n_layers=dfs.NumberOfLayers,
                    n_sigma=dfs.NumberOfSigmaLayers,
                )
            else:
                self._geometry = GeometryFM(
                    node_coordinates=nc,
                    element_table=el_table,
                    codes=codes,
                    projection_string=dfs.Projection.WKTString,
                    dfsu_type=self._type,
                    element_ids=el_ids,
                    node_ids=node_ids,
                )

        # items
        self._n_items = len(dfs.ItemInfo)
        self._items = _get_item_info(dfs.ItemInfo, list(range(self._n_items)))

        # time
        self._start_time = dfs.StartDateTime
        self._n_timesteps = dfs.NumberOfTimeSteps
        self._timestep_in_seconds = dfs.TimeStepInSeconds

        dfs.Close()

    @staticmethod
    def _get_nodes_from_source(source):
        xn = source.X
        yn = source.Y
        zn = source.Z
        nc = np.column_stack([xn, yn, zn])
        codes = np.array(list(source.Code))
        node_ids = source.NodeIds - 1
        return nc, codes, node_ids

    @staticmethod
    def _get_elements_from_source(source):
        element_table = _UnstructuredGeometry._get_element_table_from_mikecore(
            source.ElementTable
        )
        element_ids = source.ElementIds - 1
        return element_table, element_ids


class _Dfsu(_UnstructuredFile, EquidistantTimeSeries):
    def __init__(self, filename, dtype=np.float64):
        """
        Create a Dfsu object

        Parameters
        ---------
        filename: str
            dfsu or mesh filename
        dtype: np.dtype, optional
            default np.float64, valid options are np.float32, np.float64
        """
        if dtype not in [np.float32, np.float64]:
            raise ValueError("Invalid data type. Choose np.float32 or np.float64")

        super().__init__()
        self._filename = str(filename)
        self._read_header(self._filename)
        self._dtype = dtype

        # show progress bar for large files
        # if self._type == UnstructuredType.Mesh:
        #    tot_size = self.n_elements
        # else:
        #    tot_size = self.n_elements * self.n_timesteps * self.n_items
        # if tot_size > 1e6:
        #    self.show_progress = True

    @property
    def deletevalue(self):
        """File delete value"""
        return self._deletevalue

    @property
    def n_items(self):
        """Number of items"""
        return self._n_items

    @property
    def items(self):
        """List of items"""
        return self._items

    @property
    def start_time(self):
        """File start time"""
        return self._start_time

    @property
    def n_timesteps(self):
        """Number of time steps"""
        return self._n_timesteps

    @property
    def timestep(self):
        """Time step size in seconds"""
        return self._timestep_in_seconds

    @property
    def end_time(self):
        """File end time"""
        return self.start_time + timedelta(
            seconds=((self.n_timesteps - 1) * self.timestep)
        )

    def _get_spectral_data_shape(self, n_steps: int, elements):

        n_freq = self.n_frequencies
        n_dir = self.n_directions
        shape = (n_dir, n_freq)
        if n_dir == 0:
            shape = [n_freq]
        elif n_freq == 0:
            shape = [n_dir]
        if self._type == DfsuFileType.DfsuSpectral0D:
            data = np.ndarray(shape=(n_steps, *shape), dtype=self._dtype)
        elif self._type == DfsuFileType.DfsuSpectral1D:
            # node-based, FE-style
            n_nodes = self.n_nodes if elements is None else len(elements)
            data = np.ndarray(shape=(n_steps, n_nodes, *shape), dtype=self._dtype)
            shape = (*shape, self.n_nodes)
        else:
            n_elems = self.n_elements if elements is None else len(elements)
            data = np.ndarray(shape=(n_steps, n_elems, *shape), dtype=self._dtype)
            shape = (*shape, self.n_elements)

        return data, shape

    def read(self, items=None, time_steps=None, elements=None):
        """
        Read data from a dfsu file

        Parameters
        ---------
        items: list[int] or list[str], optional
            Read only selected items, by number (0-based), or by name
        time_steps: str, int or list[int], optional
            Read only selected time_steps
        elements: list[int], optional
            Read only selected element ids

        Returns
        -------
        Dataset
            A Dataset with data dimensions [t,elements]

        Examples
        --------
        >>> dfsu.read()
        <mikeio.Dataset>
        Dimensions: (9, 884)
        Time: 1985-08-06 07:00:00 - 1985-08-07 03:00:00
        Items:
        0:  Surface elevation <Surface Elevation> (meter)
        1:  U velocity <u velocity component> (meter per sec)
        2:  V velocity <v velocity component> (meter per sec)
        3:  Current speed <Current Speed> (meter per sec)
        >>> dfsu.read(time_steps="1985-08-06 12:00,1985-08-07 00:00")
        <mikeio.Dataset>
        Dimensions: (5, 884)
        Time: 1985-08-06 12:00:00 - 1985-08-06 22:00:00
        Items:
        0:  Surface elevation <Surface Elevation> (meter)
        1:  U velocity <u velocity component> (meter per sec)
        2:  V velocity <v velocity component> (meter per sec)
        3:  Current speed <Current Speed> (meter per sec)
        """

        # Open the dfs file for reading
        # self._read_dfsu_header(self._filename)
        dfs = DfsuFile.Open(self._filename)
        # time may have changes since we read the header
        # (if engine is continuously writing to this file)
        # TODO: add more checks that this is actually still the same file
        # (could have been replaced in the meantime)

        item_numbers = _valid_item_numbers(dfs.ItemInfo, items)
        items = _get_item_info(dfs.ItemInfo, item_numbers)
        n_items = len(item_numbers)

        self._n_timesteps = dfs.NumberOfTimeSteps
        time_steps = _valid_timesteps(dfs, time_steps)

        if elements is None:
            n_elems = self.n_elements
            n_nodes = self.n_nodes
            geometry = self.geometry
        else:
            n_elems = len(elements)
            geometry = self.geometry.elements_to_geometry(elements)
            if self.is_layered and items[0].name == "Z coordinate":
                node_ids, _ = self.geometry._get_nodes_and_table_for_elements(elements)
                n_nodes = len(node_ids)

        deletevalue = self.deletevalue

        data_list = []

        n_steps = len(time_steps)
        item0_is_node_based = False
        for item in range(n_items):
            # Initialize an empty data block
            if self.is_spectral:
                data, shape = self._get_spectral_data_shape(n_steps, elements=elements)
            elif item == 0 and items[item].name == "Z coordinate":
                item0_is_node_based = True
                data = np.ndarray(shape=(n_steps, n_nodes), dtype=self._dtype)
            else:
                data = np.ndarray(shape=(n_steps, n_elems), dtype=self._dtype)
            data_list.append(data)

        t_seconds = np.zeros(n_steps, dtype=float)

        for i in trange(n_steps, disable=not self.show_progress):
            it = time_steps[i]
            for item in range(n_items):

                itemdata = dfs.ReadItemTimeStep(item_numbers[item] + 1, it)
                d = itemdata.Data
                d[d == deletevalue] = np.nan

                if self.is_spectral:
                    d = np.reshape(d, newshape=shape)
                    if self._type != DfsuFileType.DfsuSpectral0D:
                        d = np.moveaxis(d, -1, 0)

                if elements is not None:
                    if item == 0 and item0_is_node_based:
                        d = d[node_ids]
                    elif self.is_spectral:
                        d = d[elements, ...]
                    else:
                        d = d[elements]

                data_list[item][i, ...] = d

            t_seconds[i] = itemdata.Time

        time = pd.to_datetime(t_seconds, unit="s", origin=self.start_time)

        dfs.Close()
        return Dataset(data_list, time, items, geometry=geometry)

    def write_header(
        self,
        filename,
        start_time=None,
        dt=None,
        items=None,
        elements=None,
        title=None,
    ):
        """Write the header of a new dfsu file (for writing huge files)

        Parameters
        -----------
        filename: str
            full path to the new dfsu file
        start_time: datetime, optional
            start datetime, default is datetime.now()
        dt: float, optional
            The time step (in seconds)
        items: list[ItemInfo], optional
        elements: list[int], optional
            write only these element ids to file
        title: str
            title of the dfsu file. Default is blank.

        Examples
        --------
        >>> msh = Mesh("foo.mesh")
        >>> n_elements = msh.n_elements
        >>> dfs = Dfsu(meshfilename)
        >>> nt = 1000
        >>> n_items = 10
        >>> items = [ItemInfo(f"Item {i+1}") for i in range(n_items)]
        >>> with dfs.write_header(outfilename, items=items) as f:
        >>>     for i in range(1, nt):
        >>>         data = []
        >>>         for i in range(n_items):
        >>>             d = np.random.random((1, n_elements))
        >>>             data.append(d)
        >>>             f.append(data)
        """

        return self.write(
            filename=filename,
            data=[],
            start_time=start_time,
            dt=dt,
            items=items,
            elements=elements,
            title=title,
            keep_open=True,
        )

    def write(
        self,
        filename,
        data,
        start_time=None,
        dt=None,
        items=None,
        elements=None,
        title=None,
        keep_open=False,
    ):
        """Write a new dfsu file

        Parameters
        -----------
        filename: str
            full path to the new dfsu file
        data: list[np.array] or Dataset
            list of matrices, one for each item. Matrix dimension: time, x
        start_time: datetime, optional, deprecated
            start datetime, default is datetime.now()
        dt: float, optional, deprecated
            The time step (in seconds)
        items: list[ItemInfo], optional, deprecated
        elements: list[int], optional
            write only these element ids to file
        title: str
            title of the dfsu file. Default is blank.
        keep_open: bool, optional
            Keep file open for appending
        """
        if self.is_spectral:
            raise ValueError("write() is not supported for spectral dfsu!")

        if start_time:
            warnings.warn(
                "setting start_time is deprecated, please supply data in the form of a Dataset",
                FutureWarning,
            )

        if dt:
            warnings.warn(
                "setting dt is deprecated, please supply data in the form of a Dataset",
                FutureWarning,
            )

        if items:
            warnings.warn(
                "setting items is deprecated, please supply data in the form of a Dataset",
                FutureWarning,
            )

        if isinstance(data, list):
            warnings.warn(
                "supplying data as a list of numpy arrays is deprecated, please supply data in the form of a Dataset",
                FutureWarning,
            )

        filename = str(filename)

        if isinstance(data, Dataset):
            items = data.items
            start_time = data.time[0]
            if dt is None and len(data.time) > 1:
                if not data.is_equidistant:
                    raise Exception(
                        "Data is not equidistant in time. Dfsu requires equidistant temporal axis!"
                    )
                dt = (data.time[1] - data.time[0]).total_seconds()
            data = data.data

        n_items = len(data)
        n_time_steps = 0
        if n_items > 0:
            n_time_steps = np.shape(data[0])[0]

        if dt is None:
            if self.timestep is None:
                dt = 1
            else:
                dt = self.timestep  # 1 # Arbitrary if there is only a single timestep

        if start_time is None:
            if self.start_time is None:
                start_time = datetime.now()
                warnings.warn(
                    f"No start time supplied. Using current time: {start_time} as start time."
                )
            else:
                start_time = self.start_time
                warnings.warn(
                    f"No start time supplied. Using start time from source: {start_time} as start time."
                )

        if items is None:
            if n_items == 0:
                raise ValueError(
                    "Number of items unknown. Add (..., items=[ItemInfo(...)]"
                )
            items = [ItemInfo(f"Item {i + 1}") for i in range(n_items)]

        if title is None:
            title = ""

        file_start_time = start_time

        # spatial subset
        if elements is None:
            geometry = self.geometry
        else:
            geometry = self.geometry.elements_to_geometry(elements)
            if (not self.is_2d) and (geometry._type == DfsuFileType.Dfsu2D):
                # redo extraction as 2d:
                # print("will redo extraction in 2d!")
                geometry = self.geometry.elements_to_geometry(
                    elements, node_layers="bottom"
                )
                if items[0].name == "Z coordinate":
                    # get rid of z-item
                    items = items[1 : (n_items + 1)]
                    n_items = n_items - 1
                    new_data = []
                    for j in range(n_items):
                        new_data.append(data[j + 1])
                    data = new_data

        # Default filetype;
        if geometry._type is None:  # == DfsuFileType.Mesh:
            # create dfs2d from mesh
            dfsu_filetype = DfsuFileType.Dfsu2D
        else:
            #    # TODO: if subset is slice...
            dfsu_filetype = geometry._type.value

        if dfsu_filetype != DfsuFileType.Dfsu2D:
            if items[0].name != "Z coordinate":
                raise Exception("First item must be z coordinates of the nodes!")

        xn = geometry.node_coordinates[:, 0]
        yn = geometry.node_coordinates[:, 1]

        # zn have to be Single precision??
        zn = geometry.node_coordinates[:, 2]

        # TODO verify this
        # elem_table = geometry.element_table
        elem_table = []
        for j in range(geometry.n_elements):
            elem_nodes = geometry.element_table[j]
            elem_nodes = [nd + 1 for nd in elem_nodes]
            elem_table.append(np.array(elem_nodes))
        elem_table = elem_table

        builder = DfsuBuilder.Create(dfsu_filetype)

        builder.SetNodes(xn, yn, zn, geometry.codes)
        builder.SetElements(elem_table)
        # builder.SetNodeIds(geometry.node_ids+1)
        # builder.SetElementIds(geometry.elements+1)

        factory = DfsFactory()
        proj = factory.CreateProjection(geometry.projection_string)
        builder.SetProjection(proj)
        builder.SetTimeInfo(file_start_time, dt)
        builder.SetZUnit(eumUnit.eumUmeter)

        if dfsu_filetype != DfsuFileType.Dfsu2D:
            builder.SetNumberOfSigmaLayers(geometry.n_sigma_layers)

        for item in items:
            if item.name != "Z coordinate":
                builder.AddDynamicItem(
                    item.name, eumQuantity.Create(item.type, item.unit)
                )

        try:
            self._dfs = builder.CreateFile(filename)
        except IOError:
            print("cannot create dfsu file: ", filename)

        deletevalue = self._dfs.DeleteValueFloat

        try:
            # Add data for all item-timesteps, copying from source
            for i in trange(n_time_steps, disable=not self.show_progress):
                for item in range(n_items):
                    d = data[item][i, :]
                    d[np.isnan(d)] = deletevalue
                    darray = d
                    self._dfs.WriteItemTimeStepNext(0, darray.astype(np.float32))
            if not keep_open:
                self._dfs.Close()
            else:
                return self

        except Exception as e:
            print(e)
            self._dfs.Close()
            os.remove(filename)

    def append(self, data: Dataset) -> None:
        """Append to a dfsu file opened with `write(...,keep_open=True)`

        Parameters
        -----------
        data: Dataset
        """

        deletevalue = self._dfs.DeleteValueFloat
        n_items = len(data)
        n_time_steps = np.shape(data[0])[0]
        for i in trange(n_time_steps, disable=not self.show_progress):
            for item in range(n_items):
                di = data[item]
                if isinstance(data, Dataset):
                    di = di.to_numpy()
                d = di[i, :]
                d[np.isnan(d)] = deletevalue
                darray = d.astype(np.float32)
                self._dfs.WriteItemTimeStepNext(0, darray)

    def close(self):
        "Finalize write for a dfsu file opened with `write(...,keep_open=True)`"
        self._dfs.Close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self._dfs.Close()

    def to_mesh(self, outfilename):
        """write object to mesh file

        Parameters
        ----------
        outfilename : str
            path to file to be written
        """
        if self.is_2d:
            # make sure element table has been constructured
            _ = self.element_table
            geometry = self.geometry
        else:
            geometry = self._geometry2d

        Mesh._geometry_to_mesh(outfilename, geometry)

    def to_dfs2(
        self,
        x0: float,
        y0: float,
        dx: float,
        dy: float,
        nx: int = 20,
        ny: int = 20,
        rotation: float = 0,
        epsg: typing.Optional[int] = None,
        interpolation_method: str = "nearest",
        filename: typing.Optional[typing.Union[str, pathlib.Path]] = None,
    ):
        """Export Dfsu to Dfs2 file.

        Export Dfsu file to a Dfs2 file with a regular 2D grid.

        Parameters
        ----------
        x0 : float
            X-coordinate of the bottom left corner of the 2D grid,
            must be in the same coordinate system as the parent Dfsu file.
        y0 : float
            Y-coordinate of the bottom left corner of the 2D grid,
            must be in the same coordinate system as the parent Dfsu file.
        dx : float
            Grid resolution in the X direction in the units of CRS defined by `epsg`.
        dy : float
            Grid resolution in the Y direction in the units of CRS defined by `epsg`.
        nx : int, optional
            Grid size in the X direction. By default it is 20.
        ny : int, optional
            Grid size in the Y direction. By default it is 20.
        rotation : float, optional
            Grid clockwise rotation in degrees. Be default it is 0.
        epsg : int, optional
            EPSG identificator of coordinate system
            in which the Dfs2 file will be created.
            If None (default), uses coordinate system of the parent Dfsu file.
        interpolation_method : str, optional
            Interpolation method, by default it is 'nearest'.
        filename : str or pathlib.Path, optional
            Path to dfs2 file to be created.
            If None (default), creates a temporary dfs2 file
            in the system temporary directory.

        Returns
        -------
        Dfs2
            mikeio Dfs2 object pointing to the file located at `filename`.

        """
        if self.is_spectral:
            raise ValueError("Method not supported for spectral dfsu!")

        # Process 'filename' argument
        if filename is None:
            filename = tempfile.NamedTemporaryFile().name + ".dfs2"
        else:
            if isinstance(filename, str):
                filename = pathlib.Path(filename)

            if isinstance(filename, pathlib.Path):
                filename = filename.resolve()
                if not filename.suffix == ".dfs2":
                    raise ValueError(
                        f"'filename' must point to a dfs2 file, "
                        f"not to '{filename.suffix}'"
                    )
            else:
                raise TypeError(
                    f"invalid type in '{type(filename)}' for the 'filename' argument, "
                    f"must be string or pathlib.Path"
                )

        # Define 2D grid in 'epsg' projection
        grid = Grid2D(
            bbox=[
                x0,
                y0,
                x0 + dx * nx,
                y0 + dy * ny,
            ],
            shape=(nx, ny),
        )
        # TODO - create rotated grid
        if rotation != 0:
            raise NotImplementedError(
                "'rotation' argument is currently not supported, "
                "grid is assumed to have its y-axis pointing at True North"
            )

        # Determine Dfsu projection
        # Convert X/Y points from Dfsu to 'epsg' projection
        # TODO - infer CRS and transform between Dfsu and Dfs2 coordinate sytems
        if epsg is not None:
            raise NotImplementedError(
                "'epsg' argument is currently not supported, "
                "coordinate system is taken from the parent Dfsu file"
            )

        # Interpolate Dfsu items to 2D grid using scipy.interpolate.griddata
        # TODO - interpolate between Dfs2 and Dfsu grids, taking into account
        # TODO - interpolation method, CRS, and grid rotation
        if interpolation_method != "nearest":
            raise NotImplementedError(
                "'interpolation_method' argument is currently not supported, "
                "interpolation is performed using nearest neighborhood method"
            )
        elem_ids, weights = self.get_2d_interpolant(
            xy=grid.xy,
            n_nearest=1,
            extrapolate=False,
            p=2,
            radius=None,
        )
        dataset = self.read(items=None, time_steps=None, elements=None)
        interpolated_dataset = self.interp2d(
            dataset,
            elem_ids=elem_ids,
            weights=weights,
            shape=(grid.ny, grid.nx),
        )

        interpolated_dataset = interpolated_dataset.flipud()

        # Write interpolated data to 'filename'
        dfs2 = Dfs2()
        dfs2.write(
            filename=str(filename),
            data=interpolated_dataset,
            start_time=dataset.time[0].to_pydatetime(),
            dt=dataset.timestep,
            items=self.items,
            dx=grid.dx,
            dy=grid.dy,
            coordinate=[
                self.projection_string,  # projection
                grid.x0,  # origin_x
                grid.y0,  # orign_y
                0,  # grid orientation - TODO account for 'rotation' argument
            ],
            title=None,  # TODO - infer it from parent Dfsu
        )

        # Return reference to the created Dfs2 file
        return Dfs2(filename=str(filename))


class Dfsu2DH(_Dfsu):
    def extract_track(self, track, items=None, method="nearest"):
        """
        Extract track data from a dfsu file

        Parameters
        ---------
        track: pandas.DataFrame
            with DatetimeIndex and (x, y) of track points as first two columns
            x,y coordinates must be in same coordinate system as dfsu
        track: str
            filename of csv or dfs0 file containing t,x,y
        items: list[int] or list[str], optional
            Extract only selected items, by number (0-based), or by name
        method: str, optional
            Spatial interpolation method ('nearest' or 'inverse_distance')
            default='nearest'

        Returns
        -------
        Dataset
            A dataset with data dimension t
            The first two items will be x- and y- coordinates of track

        Examples
        --------
        >>> ds = dfsu.extract_track(times, xy, items=['u','v'])

        >>> ds = dfsu.extract_track('track_file.dfs0')

        >>> ds = dfsu.extract_track('track_file.csv', items=0)
        """
        if self.is_spectral:
            raise ValueError("Method not supported for spectral dfsu!")

        dfs = DfsuFile.Open(self._filename)

        item_numbers = _valid_item_numbers(dfs.ItemInfo, items)
        items = _get_item_info(dfs.ItemInfo, item_numbers)
        n_items = len(item_numbers)

        self._n_timesteps = dfs.NumberOfTimeSteps
        time_steps = _valid_timesteps(dfs, time_steps=None)

        deletevalue = self.deletevalue

        if isinstance(track, str):
            filename = track
            if os.path.exists(filename):
                _, ext = os.path.splitext(filename)
                if ext == ".dfs0":
                    df = Dfs0(filename).to_dataframe()
                elif ext == ".csv":
                    df = pd.read_csv(filename, index_col=0, parse_dates=True)
                else:
                    raise ValueError(f"{ext} files not supported (dfs0, csv)")

                times = df.index
                coords = df.iloc[:, 0:2].to_numpy(copy=True)
            else:
                raise ValueError(f"{filename} does not exist")
        elif isinstance(track, Dataset):
            times = track.time
            coords = np.zeros(shape=(len(times), 2))
            coords[:, 0] = track.data[0].copy()
            coords[:, 1] = track.data[1].copy()
        else:
            assert isinstance(track, pd.DataFrame)
            times = track.index
            coords = track.iloc[:, 0:2].to_numpy(copy=True)

        assert isinstance(
            times, pd.DatetimeIndex
        ), "The index must be a pandas.DatetimeIndex"
        assert (
            times.is_monotonic_increasing
        ), "The time index must be monotonic increasing. Consider df.sort_index() before passing to extract_track()."

        data_list = []
        data_list.append(coords[:, 0])  # longitude
        data_list.append(coords[:, 1])  # latitude
        for item in range(n_items):
            # Initialize an empty data block
            data = np.empty(shape=(len(times)), dtype=self._dtype)
            data[:] = np.nan
            data_list.append(data)

        if self.is_geo:
            lon = coords[:, 0]
            lon[lon < -180] = lon[lon < -180] + 360
            lon[lon >= 180] = lon[lon >= 180] - 360
            coords[:, 0] = lon

        # track end (relative to dfsu)
        t_rel = (times - self.end_time).total_seconds()
        # largest idx for which (times - self.end_time)<=0
        tmp = np.where(t_rel <= 0)[0]
        if len(tmp) == 0:
            raise ValueError("No time overlap! Track ends before dfsu starts!")
        i_end = tmp[-1]

        # track time relative to dfsu start
        t_rel = (times - self.start_time).total_seconds()
        tmp = np.where(t_rel >= 0)[0]
        if len(tmp) == 0:
            raise ValueError("No time overlap! Track starts after dfsu ends!")
        i_start = tmp[0]  # smallest idx for which t_rel>=0

        dfsu_step = int(np.floor(t_rel[i_start] / self.timestep))  # first step

        # spatial interpolation
        n_pts = 1 if method == "nearest" else 5
        elem_ids, weights = self.get_2d_interpolant(
            coords[i_start : (i_end + 1)], n_nearest=n_pts
        )

        # initialize dfsu data arrays
        d1 = np.ndarray(shape=(n_items, self.n_elements), dtype=self._dtype)
        d2 = np.ndarray(shape=(n_items, self.n_elements), dtype=self._dtype)
        t1 = 0.0
        t2 = 0.0

        # very first dfsu time step
        step = time_steps[dfsu_step]
        for item in range(n_items):
            itemdata = dfs.ReadItemTimeStep(item_numbers[item] + 1, step)
            t2 = itemdata.Time - 1e-10
            # d = to_numpy(itemdata.Data)
            d = itemdata.Data
            d[d == deletevalue] = np.nan
            d2[item, :] = d

        def is_EOF(step):
            return step >= self.n_timesteps

        # loop over track points
        for i_interp, i in enumerate(
            trange(i_start, i_end + 1, disable=not self.show_progress)
        ):
            t_rel[i]  # time of point relative to dfsu start

            read_next = t_rel[i] > t2

            while (read_next == True) and (not is_EOF(dfsu_step + 1)):
                dfsu_step = dfsu_step + 1

                # swap new to old
                d1, d2 = d2, d1
                t1, t2 = t2, t1

                step = time_steps[dfsu_step]
                for item in range(n_items):
                    itemdata = dfs.ReadItemTimeStep(item_numbers[item] + 1, step)
                    t2 = itemdata.Time
                    # d = to_numpy(itemdata.Data)
                    d = itemdata.Data
                    d[d == deletevalue] = np.nan
                    d2[item, :] = d

                read_next = t_rel[i] > t2

            if (read_next == True) and (is_EOF(dfsu_step)):
                # cannot read next - no more timesteps in dfsu file
                continue

            w = (t_rel[i] - t1) / self.timestep  # time-weight
            eid = elem_ids[i_interp]
            if np.any(eid > 0):
                dati = (1 - w) * np.dot(d1[:, eid], weights[i_interp])
                dati = dati + w * np.dot(d2[:, eid], weights[i_interp])
            else:
                dati = np.empty(shape=n_items, dtype=self._dtype)
                dati[:] = np.nan

            for item in range(n_items):
                data_list[item + 2][i] = dati[item]

        dfs.Close()

        items_out = []
        if self.is_geo:
            items_out.append(ItemInfo("Longitude"))
            items_out.append(ItemInfo("Latitude"))
        else:
            items_out.append(ItemInfo("x"))
            items_out.append(ItemInfo("y"))
        for item in items:
            items_out.append(item)

        return Dataset(data_list, times, items_out)


class DfsuLayered(_Dfsu):
    @wraps(GeometryFMLayered.to_2d_geometry)
    def to_2d_geometry(self):
        return self.geometry2d

    @property
    def geometry2d(self):
        """The 2d geometry for a 3d object"""
        return self._geometry2d

    @property
    def n_layers(self):
        """Maximum number of layers"""
        return self.geometry._n_layers

    @property
    def n_sigma_layers(self):
        """Number of sigma layers"""
        return self.geometry.n_sigma_layers

    @property
    def n_z_layers(self):
        """Maximum number of z-layers"""
        if self.n_layers is None:
            return None
        return self.n_layers - self.n_sigma_layers

    @property
    def e2_e3_table(self):
        """The 2d-to-3d element connectivity table for a 3d object"""
        if self.n_layers is None:
            print("Object has no layers: cannot return e2_e3_table")
            return None
        return self.geometry.e2_e3_table

    @property
    def elem2d_ids(self):
        """The associated 2d element id for each 3d element"""
        if self.n_layers is None:
            raise InvalidGeometry("Object has no layers: cannot return elem2d_ids")
        return self.geometry.elem2d_ids

    @property
    def layer_ids(self):
        """The layer number (0=bottom, 1, 2, ...) for each 3d element"""
        if self.n_layers is None:
            raise InvalidGeometry("Object has no layers: cannot return layer_ids")
        return self.geometry.layer_ids

    @property
    def top_elements(self):
        """List of 3d element ids of surface layer"""
        if self.n_layers is None:
            print("Object has no layers: cannot find top_elements")
            return None
        return self.geometry.top_elements

    @property
    def n_layers_per_column(self):
        """List of number of layers for each column"""
        if self.n_layers is None:
            print("Object has no layers: cannot find n_layers_per_column")
            return None
        return self.geometry.n_layers_per_column

    @property
    def bottom_elements(self):
        """List of 3d element ids of bottom layer"""
        if self.n_layers is None:
            print("Object has no layers: cannot find bottom_elements")
            return None
        return self.geometry.bottom_elements

    @wraps(GeometryFMLayered.get_layer_elements)
    def get_layer_elements(self, layer):
        if self.n_layers is None:
            raise InvalidGeometry("Object has no layers: cannot get_layer_elements")
        return self.geometry.get_layer_elements(layer)


class Dfsu2DV(DfsuLayered):
    def plot_vertical_profile(
        self, values, time_step=None, cmin=None, cmax=None, label="", **kwargs
    ):
        """
        Plot unstructured vertical profile

        Parameters
        ----------
        values: np.array
            value for each element to plot
        timestep: int, optional
            the timestep that fits with the data to get correct vertical
            positions, default: use static vertical positions
        cmin: real, optional
            lower bound of values to be shown on plot, default:None
        cmax: real, optional
            upper bound of values to be shown on plot, default:None
        title: str, optional
            axes title
        label: str, optional
            colorbar label
        cmap: matplotlib.cm.cmap, optional
            colormap, default viridis
        figsize: (float, float), optional
            specify size of figure
        ax: matplotlib.axes, optional
            Adding to existing axis, instead of creating new fig

        Returns
        -------
        <matplotlib.axes>
        """
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection

        if isinstance(values, DataArray):
            values = values.to_numpy()

        nc = self.node_coordinates
        x_coordinate = np.hypot(nc[:, 0], nc[:, 1])
        if time_step is None:
            y_coordinate = nc[:, 2]
        else:
            y_coordinate = self.read()[0].to_numpy()[time_step, :]

        elements = self._Get_2DVertical_elements()

        # plot in existing or new axes?
        if "ax" in kwargs:
            ax = kwargs["ax"]
        else:
            figsize = None
            if "figsize" in kwargs:
                figsize = kwargs["figsize"]
            _, ax = plt.subplots(figsize=figsize)

        yz = np.c_[x_coordinate, y_coordinate]
        verts = yz[elements]

        if "cmap" in kwargs:
            cmap = kwargs["cmap"]
        else:
            cmap = "jet"
        pc = PolyCollection(verts, cmap=cmap)

        if cmin is None:
            cmin = np.nanmin(values)
        if cmax is None:
            cmax = np.nanmax(values)
        pc.set_clim(cmin, cmax)

        plt.colorbar(pc, ax=ax, label=label, orientation="vertical")
        pc.set_array(values)

        if "edge_color" in kwargs:
            edge_color = kwargs["edge_color"]
        else:
            edge_color = None
        pc.set_edgecolor(edge_color)

        ax.add_collection(pc)
        ax.autoscale()

        if "title" in kwargs:
            ax.set_title(kwargs["title"])

        return ax

    def _Get_2DVertical_elements(self):
        if (self._type == DfsuFileType.DfsuVerticalProfileSigmaZ) or (
            self._type == DfsuFileType.DfsuVerticalProfileSigma
        ):
            elements = [
                list(self.geometry.element_table[i])
                for i in range(len(list(self.geometry.element_table)))
            ]
            return np.asarray(elements)  # - 1


class Dfsu3D(DfsuLayered):
    def find_nearest_profile_elements(self, x, y):
        """Find 3d elements of profile nearest to (x,y) coordinates

        Parameters
        ----------
        x : float
            x-coordinate of point
        y : float
            y-coordinate of point

        Returns
        -------
        np.array(int)
            element ids of vertical profile
        """
        if self.is_2d:
            raise InvalidGeometry("Object is 2d. Cannot get_nearest_profile")
        else:
            elem2d, _ = self.geometry._find_n_nearest_2d_elements(x, y)
            elem3d = self.geometry.e2_e3_table[elem2d]
            return elem3d

    def extract_surface_elevation_from_3d(
        self, filename=None, time_steps=None, n_nearest=4
    ):
        """
        Extract surface elevation from a 3d dfsu file (based on zn)
        to a new 2d dfsu file with a surface elevation item.

        Parameters
        ---------
        filename: str
            Output file name
        time_steps: str, int or list[int], optional
            Extract only selected time_steps
        n_nearest: int, optional
            number of points for spatial interpolation (inverse_distance), default=4

        Examples
        --------
        >>> dfsu.extract_surface_elevation_from_3d('ex_surf.dfsu', time_steps='2018-1-1,2018-2-1')
        """
        # validate input
        assert (
            self._type == DfsuFileType.Dfsu3DSigma
            or self._type == DfsuFileType.Dfsu3DSigmaZ
        )
        assert n_nearest > 0
        time_steps = _valid_timesteps(self._source, time_steps)

        # make 2d nodes-to-elements interpolator
        top_el = self.top_elements
        geom = self.geometry.elements_to_geometry(top_el, node_layers="top")
        xye = geom.element_coordinates[:, 0:2]
        xyn = geom.node_coordinates[:, 0:2]
        tree2d = cKDTree(xyn)
        dist, node_ids = tree2d.query(xye, k=n_nearest)
        if n_nearest == 1:
            weights = None
        else:
            weights = get_idw_interpolant(dist)

        # read zn from 3d file and interpolate to element centers
        ds = self.read(items=0, time_steps=time_steps)  # read only zn
        node_ids_surf, _ = self.geometry._get_nodes_and_table_for_elements(
            top_el, node_layers="top"
        )
        zn_surf = ds.data[0][:, node_ids_surf]  # surface
        surf2d = interp2d(zn_surf, node_ids, weights)

        # create output
        items = [ItemInfo(EUMType.Surface_Elevation)]
        ds2 = Dataset([surf2d], ds.time, items, geometry=geom)
        if filename is None:
            return ds2
        else:
            title = "Surface extracted from 3D file"
            self.write(filename, ds2, elements=top_el, title=title)


class DfsuSpectral(_Dfsu):
    def plot_spectrum(
        self,
        spectrum,
        plot_type="contourf",
        title=None,
        label=None,
        cmap="Reds",
        vmin=1e-5,
        vmax=None,
        r_as_periods=True,
        rmin=None,
        rmax=None,
        levels=None,
        figsize=(7, 7),
        add_colorbar=True,
    ):
        """
        Plot spectrum in polar coordinates

        Parameters
        ----------
        spectrum: np.array
            spectral values as 2d array with dimensions: directions, frequencies
        plot_type: str, optional
            type of plot: 'contour', 'contourf', 'patch', 'shaded',
            by default: 'contourf'
        title: str, optional
            axes title
        label: str, optional
            colorbar label (or title if contour plot)
        cmap: matplotlib.cm.cmap, optional
            colormap, default Reds
        vmin: real, optional
            lower bound of values to be shown on plot, default: 1e-5
        vmax: real, optional
            upper bound of values to be shown on plot, default:None
        r_as_periods: bool, optional
            show radial axis as periods instead of frequency, default: True
        rmin: float, optional
            mininum frequency/period to be shown, default: None
        rmax: float, optional
            maximum frequency/period to be shown, default: None
        levels: int, list(float), optional
            for contour plots: how many levels, default:10
            or a list of discrete levels e.g. [0.03, 0.04, 0.05]
        figsize: (float, float), optional
            specify size of figure, default (7, 7)
        add_colorbar: bool, optional
            Add colorbar to plot, default True

        Returns
        -------
        <matplotlib.axes>

        Examples
        --------
        >>> dfs = Dfsu("area_spectrum.dfsu")
        >>> ds = dfs.read(items="Energy density")
        >>> spectrum = ds[0][0, 0, :, :] # first timestep, element 0
        >>> dfs.plot_spectrum(spectrum, plot_type="patch")

        >>> dfs.plot_spectrum(spectrum, rmax=9, title="Wave spectrum T<9s")
        """
        if isinstance(spectrum, DataArray):
            spectrum = spectrum.to_numpy()
        # TODO move this to specialized class e.g. DfsuSpectral

        if self.n_directions == 0 or self.n_frequencies == 0:
            raise ValueError("plot_spectrum() is only supported for spectral data")

        import matplotlib.pyplot as plt

        dirs = np.radians(self.directions)
        freq = self.frequencies

        inverse_r = r_as_periods
        if inverse_r:
            freq = 1.0 / np.flip(freq)
            spectrum = np.fliplr(spectrum)

        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(111, polar=True)
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location("N")

        ddir = dirs[1] - dirs[0]

        def is_circular(dir):
            dir_diff = np.mod(dir[0], 2 * np.pi) - np.mod(dir[-1] + ddir, 2 * np.pi)
            return np.abs(dir_diff) < 1e-6

        if is_circular(dirs):
            # append last directional slice at the end
            dirs = np.append(dirs, dirs[-1] + ddir)
            spectrum = np.concatenate((spectrum, spectrum[0:1, :]), axis=0)

        # up-sample directions
        if plot_type in ("shaded", "contour", "contourf"):
            # more smoother plotting
            factor = 4
            dir2 = np.linspace(dirs[0], dirs[-1], (len(dirs) - 1) * factor + 1)
            spec2 = np.zeros(shape=(len(dir2), len(freq)))
            for j in range(len(freq)):
                spec2[:, j] = np.interp(dir2, dirs, spectrum[:, j])
            dirs = dir2.copy()
            spectrum = spec2.copy()

        if vmin is None:
            vmin = np.nanmin(spectrum)
        if vmax is None:
            vmax = np.nanmax(spectrum)
        if levels is None:
            levels = 10
            n_levels = 10
        if np.isscalar(levels):
            n_levels = levels
            levels = np.linspace(vmin, vmax, n_levels)

        if plot_type != "shaded":
            spectrum[spectrum < vmin] = np.nan

        if plot_type == "contourf":
            colorax = ax.contourf(
                dirs, freq, spectrum.T, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax
            )
        elif plot_type == "contour":
            colorax = ax.contour(
                dirs, freq, spectrum.T, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax
            )
            # ax.clabel(colorax, fmt="%1.2f", inline=1, fontsize=9)
            if label is not None:
                ax.set_title(label)

        elif plot_type in ("patch", "shaded", "box"):
            shading = "gouraud" if plot_type == "shaded" else "auto"
            colorax = ax.pcolormesh(
                dirs,
                freq,
                spectrum.T,
                shading=shading,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            ax.grid("on")
        else:
            raise ValueError(
                f"plot_type '{plot_type}' not supported (contour, contourf, patch, shaded)"
            )

        # TODO: optional
        ax.set_thetagrids(
            [0.0, 45, 90.0, 135, 180.0, 225, 270.0, 315],
            labels=["N", "N-E", "E", "S-E", "S", "S-W", "W", "N-W"],
        )

        # if r_axis_as_periods:
        #     Ts = [8, 6, 5, 4, 3.5, 3, 2.5, 2]
        #     ax.set_rgrids(
        #         1.0 / np.array(Ts),
        #         labels=["8s", "6s", "5s", "4s", "3.5s", "3s", "2.5s", "2s"],
        #     )
        #     # , fontsize=12, angle=180)

        # ax.grid(True, which='minor', axis='both', linestyle='-', color='0.8')
        # ax.set_xticks(dfs.directions, minor=True);

        if rmin is not None:
            ax.set_rmin(rmin)
        if rmax is not None:
            ax.set_rmax(rmax)

        if add_colorbar:
            cbar = fig.colorbar(colorax)
            if label is None:
                label = "Energy Density [m*m/Hz/deg]"
            cbar.set_label(label, rotation=270)
            cbar.ax.get_yaxis().labelpad = 30

        if title is not None:
            ax.set_title(title)

        return ax

    def calc_Hm0_from_spectrum(self, spectrum, tail=True):
        """Calculate significant wave height (Hm0) from spectrum

        Parameters
        ----------
        spectrum : np.ndarray
            frequency or direction-frequency spectrum
        tail : bool, optional
            Should a parametric spectral tail be added in the computations? by default True

        Returns
        -------
        np.ndarray
            significant wave height values
        """
        # if not self.is_spectral:
        #    raise ValueError("Method only supported for spectral dfsu!")

        m0 = self._calc_m0_from_spectrum(
            spectrum, self.frequencies, self.directions, tail, m0_only=True
        )
        return 4 * np.sqrt(m0)

    @staticmethod
    def _calc_m0_from_spectrum(spec, f, dir=None, tail=True, m0_only=False):
        if f is None:
            raise ValueError(
                "Moments cannot be calculated because dfsu has no frequency axis"
            )
        df = DfsuSpectral._f_to_df(f)

        if dir is None:
            ee = spec
        else:
            nd = len(dir)
            dtheta = (dir[-1] - dir[0]) / (nd - 1)
            ee = np.sum(spec, axis=-2) * dtheta

        m0 = np.dot(ee, df)
        if tail:
            m0 = m0 + ee[..., -1] * f[-1] * 0.25
        return m0

    @staticmethod
    def _f_to_df(f):
        """Frequency bins for equidistant or logrithmic frequency axis"""
        if np.isclose(np.diff(f).min(), np.diff(f).max()):
            # equidistant frequency bins
            return (f[1] - f[0]) * np.ones_like(f)
        else:
            # logarithmic frequency bins
            freq_factor = f[1] / f[0]
            fm1 = np.insert(f, 0, f[0] / freq_factor)
            fp1 = np.append(f, f[-1] * freq_factor)
            return 0.5 * (np.diff(fm1) + np.diff(fp1))


class Mesh(_UnstructuredFile):
    """
    The Mesh class is initialized with a mesh or a dfsu file.

    Parameters
    ---------
    filename: str
        dfsu or mesh filename

    Examples
    --------

    >>> msh = Mesh("../tests/testdata/odense_rough.mesh")
    >>> msh
    Number of elements: 654
    Number of nodes: 399
    Projection: UTM-33

    """

    def __init__(self, filename):
        super().__init__()
        self._filename = filename
        self._read_header(filename)
        self._n_timesteps = None
        self._n_items = None
        self._type = None  # DfsuFileType.Mesh

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
            elem_table = _UnstructuredGeometry._element_table_to_mikecore(
                geometry.element_table
            )

        nc = geometry.node_coordinates
        builder.SetNodes(nc[:, 0], nc[:, 1], nc[:, 2], geometry.codes)

        builder.SetElements(elem_table)
        builder.SetProjection(geometry.projection_string)
        builder.SetEumQuantity(quantity)

        newMesh = builder.CreateMesh()
        newMesh.Write(outfilename)

    @wraps(GeometryFM.plot_boundary_nodes)
    def plot_boundary_nodes(self, boundary_names=None, figsize=None, ax=None):
        return self.geometry.plot_boundary_nodes(boundary_names, figsize, ax)

    @staticmethod
    def _geometry_to_mesh(outfilename, geometry):

        builder = MeshBuilder()

        nc = geometry.node_coordinates
        builder.SetNodes(nc[:, 0], nc[:, 1], nc[:, 2], geometry.codes)
        # builder.SetNodeIds(geometry.node_ids+1)
        # builder.SetElementIds(geometry.elements+1)
        builder.SetElements(
            _UnstructuredGeometry._element_table_to_mikecore(geometry.element_table)
        )
        builder.SetProjection(geometry.projection_string)
        quantity = eumQuantity.Create(EUMType.Bathymetry, EUMUnit.meter)
        builder.SetEumQuantity(quantity)
        newMesh = builder.CreateMesh()
        newMesh.Write(outfilename)
