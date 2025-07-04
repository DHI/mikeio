from dataclasses import dataclass
from typing import Any, Literal, Sequence

from numpy.typing import NDArray
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure
from matplotlib.tri import Triangulation
import numpy as np
from scipy.sparse import csr_matrix

from ._distance import relative_cumulative_distance


MESH_COL = "0.95"
MESH_COL_DARK = "0.6"


@dataclass
class Polygon:
    xy: NDArray

    @property
    def area(self) -> float:
        return (
            np.dot(self.xy[:, 1], np.roll(self.xy[:, 0], 1))
            - np.dot(self.xy[:, 0], np.roll(self.xy[:, 1], 1))
        ) * 0.5


@dataclass
class BoundaryPolygons:
    exteriors: list[Polygon]
    interiors: list[Polygon]

    @property
    def lines(self) -> list[Polygon]:
        return self.exteriors + self.interiors

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Test if a list of points are contained by mesh.

        Parameters
        ----------
        points : array-like n-by-2
            x,y-coordinates of n points to be tested

        Returns
        -------
        bool array
            True for points inside, False otherwise

        """
        import matplotlib.path as mp  # type: ignore

        exterior = self.exteriors[0]
        cnts = mp.Path(exterior.xy).contains_points(points)

        if len(self.exteriors) > 1:
            # in case of several dis-joint outer domains
            for exterior in self.exteriors[1:]:
                in_domain = mp.Path(exterior.xy).contains_points(points)
                cnts = np.logical_or(cnts, in_domain)

        # subtract any holes
        for interior in self.interiors:
            in_hole = mp.Path(interior.xy).contains_points(points)
            cnts = np.logical_and(cnts, ~in_hole)

        return cnts


def _plot_map(
    node_coordinates: np.ndarray,
    element_table: np.ndarray,
    element_coordinates: np.ndarray,
    boundary_polylines: list[Polygon],
    projection: str = "",
    z: np.ndarray | None = None,
    plot_type: Literal[
        "patch", "mesh_only", "shaded", "contour", "contourf", "outline_only"
    ] = "patch",
    title: str | None = None,
    label: str | None = None,
    cmap: Colormap | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    levels: int | Sequence[float] | None = None,
    n_refinements: int = 0,
    show_mesh: bool = False,
    show_outline: bool = True,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
    add_colorbar: bool = True,
) -> Axes:
    """Plot unstructured data and/or mesh, mesh outline.

    Parameters
    ----------
    node_coordinates: np.array
        node coordinates
    element_table: np.array
        element table
    element_coordinates: np.array
        element coordinates
    boundary_polylines: BoundaryPolylines,
        boundary polylines
    projection: str, optional
        projection type, default: ""
    z: np.array or a Dataset with a single item, optional
        value for each element to plot, default bathymetry
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
    >>> ds = dfs.read(items="Surface elevation", time=0)
    >>> ds.shape
    (1, 884)
    >>> ds.n_items
    1
    >>> dfs.plot(z=ds) # plot surface elevation

    """
    import matplotlib.pyplot as plt
    import matplotlib

    VALID_PLOT_TYPES = (
        "mesh_only",
        "outline_only",
        "contour",
        "contourf",
        "patch",
        "shaded",
    )
    if plot_type not in VALID_PLOT_TYPES:
        ok_list = ", ".join(VALID_PLOT_TYPES)
        raise Exception(f"plot_type {plot_type} unknown! ({ok_list})")

    cmap = cmap or matplotlib.colormaps["viridis"]

    nc = node_coordinates
    ec = element_coordinates

    if ((vmin is not None) or (vmax is not None)) and (
        levels is not None and not np.isscalar(levels)
    ):
        raise ValueError(
            "vmin/vmax cannot be provided together with non-integer levels"
        )

    # plot in existing or new axes?
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # set aspect ratio
    __set_aspect_ratio(ax, nc, projection)
    _set_xy_label_by_projection(ax, projection)

    if plot_type == "outline_only":
        __plot_outline_only(ax, boundary_polylines)
        return ax

    if plot_type == "mesh_only":
        __plot_mesh_only(ax, nc, element_table)
        return ax

    # At this point we are sure that we are plotting some data, at least bathymetry
    if z is None:
        z = ec[:, 2]
        label = label or "Bathymetry (m)"

    assert len(z) == ec.shape[0]

    label = label or ""

    vmin, vmax, cmap, cmap_norm, cmap_ScMappable, levels = __set_colormap_levels(
        cmap, vmin, vmax, levels, z
    )
    cbar_extend = __cbar_extend(z, vmin, vmax)

    if plot_type == "patch":
        fig_obj: Any = __plot_patch(
            ax, nc, element_table, show_mesh, cmap, cmap_norm, z, vmin, vmax
        )

    else:
        # do node-based triangular plot
        mesh_linewidth = 0.0
        if show_mesh and __is_tri_only(element_table):
            mesh_linewidth = 0.4
            n_refinements = 0
        triang, zn = __get_tris(nc, element_table, ec, z, n_refinements)

        if plot_type == "shaded":
            ax.triplot(triang, lw=mesh_linewidth, color=MESH_COL)
            fig_obj = ax.tripcolor(
                triang,
                zn,
                edgecolors="face",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                linewidths=0.3,
                shading="gouraud",
            )

        elif plot_type == "contour":
            ax.triplot(triang, lw=mesh_linewidth, color=MESH_COL_DARK)
            fig_obj = ax.tricontour(
                triang,
                zn,
                levels=levels,
                linewidths=[1.2],
                cmap=cmap,
                norm=cmap_norm,
            )
            ax.clabel(fig_obj, fmt="%1.2f", inline=1, fontsize=9)
            ax.set_title(label)
            add_colorbar = False

        elif plot_type == "contourf":
            ax.triplot(triang, lw=mesh_linewidth, color=MESH_COL)
            fig_obj = ax.tricontourf(
                triang,
                zn,
                levels=levels,
                cmap=cmap,
                norm=cmap_norm,
                extend=cbar_extend,
                vmin=vmin,
                vmax=vmax,
            )

        if show_mesh and (not __is_tri_only(element_table)):
            __add_non_tri_mesh(ax, nc, element_table, plot_type)

    if show_outline:
        __add_outline(ax, boundary_polylines)

    if add_colorbar:
        __add_colorbar(ax, cmap_ScMappable, fig_obj, label, levels, cbar_extend)

    __set_plot_limits(ax, nc)

    if title:
        ax.set_title(title)

    return ax


def __set_colormap_levels(
    cmap: Colormap | str,
    vmin: float | None,
    vmax: float | None,
    levels: int | Sequence[float] | np.ndarray | None,
    z: np.ndarray,
) -> tuple[float, float, Colormap, Normalize, ScalarMappable, np.ndarray]:
    import matplotlib
    import matplotlib.cm as cm
    import matplotlib.colors as mplc

    vmin = vmin or np.nanmin(z)
    vmax = vmax or np.nanmax(z)

    if vmin == vmax:
        vmin = vmin - 0.1
        vmax = vmin + 0.2

    cmap_norm = None
    cmap_ScMappable = None
    if levels is not None:
        if np.isscalar(levels):
            n_levels = levels
            levels = np.linspace(vmin, vmax, n_levels)  # type: ignore
        else:
            n_levels = len(levels)  # type: ignore
            vmin = min(levels)  # type: ignore
            vmax = max(levels)  # type: ignore

        levels = np.array(levels)

        if isinstance(cmap, str):
            cmap = matplotlib.colormaps[cmap]
        cmap_norm = mplc.BoundaryNorm(levels, cmap.N)
        cmap_ScMappable = cm.ScalarMappable(cmap=cmap, norm=cmap_norm)

    if levels is None:
        levels = np.linspace(vmin, vmax, 10)

    return vmin, vmax, cmap, cmap_norm, cmap_ScMappable, levels  # type: ignore


def __set_plot_limits(ax: Axes, nc: np.ndarray) -> None:
    xmin, xmax = nc[:, 0].min(), nc[:, 0].max()
    ymin, ymax = nc[:, 1].min(), nc[:, 1].max()

    xybuf = 6e-3 * (xmax - xmin)
    ax.set_xlim(xmin - xybuf, xmax + xybuf)
    ax.set_ylim(ymin - xybuf, ymax + xybuf)


def __plot_mesh_only(ax: Axes, nc: np.ndarray, element_table: np.ndarray) -> None:
    from matplotlib.collections import PatchCollection

    patches = _to_polygons(nc, element_table)
    fig_obj = PatchCollection(
        patches, edgecolor=MESH_COL_DARK, facecolor="none", linewidths=0.3
    )
    ax.add_collection(fig_obj)


def __plot_outline_only(ax: Axes, boundary_polylines: list[Polygon]) -> Axes:
    __add_outline(ax, boundary_polylines)
    return ax


def __plot_patch(
    ax: Axes,
    nc: np.ndarray,
    element_table: np.ndarray,
    show_mesh: bool,
    cmap: Colormap,
    cmap_norm: Normalize,
    z: np.ndarray,
    vmin: float,
    vmax: float,
) -> PatchCollection:
    patches = _to_polygons(nc, element_table)

    if show_mesh:
        edgecolor = MESH_COL
        linewidth = 0.4
    else:
        edgecolor = "face"
        linewidth = None

    fig_obj = PatchCollection(
        patches,
        cmap=cmap,
        norm=cmap_norm,
        edgecolor=edgecolor,
        linewidths=linewidth,
    )

    fig_obj.set_array(z)
    fig_obj.set_clim(vmin, vmax)
    ax.add_collection(fig_obj)

    return fig_obj


def __get_tris(
    nc: np.ndarray,
    element_table: np.ndarray,
    ec: np.ndarray,
    z: np.ndarray,
    n_refinements: int,
) -> tuple[Triangulation, np.ndarray]:
    import matplotlib.tri as tri

    elem_table, ec, z = __create_tri_only_element_table(nc, element_table, ec, data=z)
    triang = tri.Triangulation(nc[:, 0], nc[:, 1], elem_table)

    zn = _get_node_centered_data(nc, elem_table, ec, z)

    if n_refinements > 0:
        # TODO: refinements doesn't seem to work for 3d files?
        refiner = tri.UniformTriRefiner(triang)
        triang, zn = refiner.refine_field(zn, subdiv=n_refinements)

    return triang, zn


def __add_colorbar(
    ax: Axes,
    cmap_ScMappable: ScalarMappable,
    fig_obj: Figure,
    label: str,
    levels: np.ndarray,
    cbar_extend: str,
) -> None:
    from mpl_toolkits.axes_grid1 import make_axes_locatable  # type: ignore
    import matplotlib.pyplot as plt

    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    cmap_sm = cmap_ScMappable if cmap_ScMappable else fig_obj

    plt.colorbar(
        cmap_sm,  # type: ignore
        label=label,
        cax=cax,
        ticks=levels,
        boundaries=levels,
        extend=cbar_extend,
    )


def __set_aspect_ratio(ax: Axes, nc: np.ndarray, projection: str) -> None:
    is_geo = projection == "LONG/LAT"
    if is_geo:
        mean_lat = np.mean(nc[:, 1])
        ax.set_aspect(1.0 / np.cos(np.pi * mean_lat / 180))
    else:
        ax.set_aspect("equal")


def __add_non_tri_mesh(
    ax: Axes, nc: np.ndarray, element_table: np.ndarray, plot_type: str
) -> None:
    # if mesh is not tri only, we need to add it manually on top
    from matplotlib.collections import PatchCollection

    patches = _to_polygons(nc, element_table)
    mesh_linewidth = 0.4
    if plot_type == "contour":
        mesh_col = MESH_COL_DARK
    else:
        mesh_col = MESH_COL
    p = PatchCollection(
        patches,
        edgecolor=mesh_col,
        facecolor="none",
        linewidths=mesh_linewidth,
    )
    ax.add_collection(p)


def __add_outline(ax: Axes, boundary_polylines: list[Polygon]) -> None:
    for line in boundary_polylines:
        ax.plot(*line.xy.T, color="0.4", linewidth=1.2)


def _set_xy_label_by_projection(ax: Axes, projection: str) -> None:
    if (not projection) or projection == "NON-UTM":
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
    elif projection == "LONG/LAT":
        ax.set_xlabel("Longitude [degrees]")
        ax.set_ylabel("Latitude [degrees]")
    else:
        ax.set_xlabel("Easting [m]")
        ax.set_ylabel("Northing [m]")


def __is_tri_only(element_table: np.ndarray) -> bool:
    return max([len(el) for el in element_table]) == 3


def _to_polygons(node_coordinates: np.ndarray, element_table: np.ndarray) -> list[Any]:
    from matplotlib.patches import Polygon

    polygons = []

    for j in range(len(element_table)):
        nodes = element_table[j]
        pcoords = np.empty([len(nodes), 2])
        for i in range(len(nodes)):
            nidx = nodes[i]
            pcoords[i, :] = node_coordinates[nidx, 0:2]

        polygon = Polygon(pcoords, closed=True)
        polygons.append(polygon)
    return polygons


def _create_node_element_matrix(
    element_table: np.ndarray, num_nodes: int
) -> csr_matrix:
    row_ind = element_table.ravel()
    col_ind = np.repeat(np.arange(element_table.shape[0]), element_table.shape[1])
    data = np.ones(len(row_ind), dtype=int)
    connectivity_matrix = csr_matrix(
        (data, (row_ind, col_ind)), shape=(num_nodes, element_table.shape[0])
    )
    return connectivity_matrix


def _get_node_centered_data(
    node_coordinates: np.ndarray,
    element_table: np.ndarray,
    element_coordinates: np.ndarray,
    data: np.ndarray,
    extrapolate: bool = True,
) -> np.ndarray:
    """convert cell-centered data to node-centered by pseudo-laplacian method."""
    nc = node_coordinates
    elem_table, ec, data = __create_tri_only_element_table(
        nc, element_table, element_coordinates, data
    )
    connectivity_matrix = _create_node_element_matrix(elem_table, nc.shape[0])

    node_centered_data = np.zeros(shape=nc.shape[0])
    for n in range(connectivity_matrix.shape[0]):
        item = connectivity_matrix.getrow(n).indices
        I = ec[item][:, :2] - nc[n][:2]
        I2 = (I**2).sum(axis=0)
        Ixy = (I[:, 0] * I[:, 1]).sum(axis=0)
        lamb = I2[0] * I2[1] - Ixy**2
        omega = np.zeros(1)
        if lamb > 1e-10 * (I2[0] * I2[1]):
            # Standard case - Pseudo
            lambda_x = (Ixy * I[:, 1] - I2[1] * I[:, 0]) / lamb
            lambda_y = (Ixy * I[:, 0] - I2[0] * I[:, 1]) / lamb
            omega = 1.0 + lambda_x * I[:, 0] + lambda_y * I[:, 1]
            if not extrapolate:
                omega[np.where(omega > 2)] = 2
                omega[np.where(omega < 0)] = 0
        if omega.sum() > 0:
            node_centered_data[n] = np.sum(omega * data[item]) / np.sum(omega)
        else:
            # We did not succeed using pseudo laplace procedure, use inverse distance instead
            InvDis = [
                1 / np.hypot(case[0], case[1]) for case in ec[item][:, :2] - nc[n][:2]
            ]
            node_centered_data[n] = np.sum(InvDis * data[item]) / np.sum(InvDis)

    return node_centered_data


def __create_tri_only_element_table(
    node_coordinates: np.ndarray,
    element_table: np.ndarray,
    element_coordinates: np.ndarray,
    data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert quad/tri mesh to pure tri-mesh."""
    if __is_tri_only(element_table):
        # already tri-only? just convert to 2d array
        return np.stack(element_table), element_coordinates, data  # type: ignore

    ec = element_coordinates.copy()

    elem_table = [list(element_table[i]) for i in range(len(element_table))]
    tmp_elmnt_nodes = elem_table.copy()
    for el, item in enumerate(tmp_elmnt_nodes):
        if len(item) == 4:
            elem_table.pop(el)  # remove quad element

            # insert two new tri elements in table
            elem_table.insert(el, item[:3])
            tri2_nodes = [item[i] for i in [2, 3, 0]]
            elem_table.append(tri2_nodes)

            # new center coordinates for new tri-elements
            ec[el] = node_coordinates[item[:3]].mean(axis=1)
            tri2_ec = node_coordinates[tri2_nodes].mean(axis=1)
            ec = np.append(ec, tri2_ec.reshape(1, -1), axis=0)

            # use same data in two new tri elements
            data = np.append(data, data[el])

    return np.asarray(elem_table), ec, data


def __cbar_extend(
    calc_data: np.ndarray | None, vmin: float | None, vmax: float | None
) -> str:
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


def _plot_vertical_profile(
    node_coordinates: np.ndarray,
    element_table: np.ndarray,
    values: np.ndarray,
    zn: np.ndarray | None = None,
    is_geo: bool = False,
    cmin: float | None = None,
    cmax: float | None = None,
    label: str = "",
    add_colorbar: bool = True,
    title: str | None = None,
    cmap: str | Colormap | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,
) -> Axes:
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    nc = node_coordinates
    s_coordinate = relative_cumulative_distance(nc, is_geo=is_geo)
    z_coordinate = nc[:, 2] if zn is None else zn

    elements = _Get_2DVertical_elements(element_table)

    # plot in existing or new axes?
    if ax is None:
        figsize = None
        if "figsize" in kwargs:
            figsize = kwargs["figsize"]
        _, ax = plt.subplots(figsize=figsize)

    sz = np.c_[s_coordinate, z_coordinate]
    verts = sz[elements]

    if cmap is None:
        cmap = "jet"
    pc = PolyCollection(verts, cmap=cmap)

    if cmin is None:
        cmin = np.nanmin(values)
    if cmax is None:
        cmax = np.nanmax(values)
    pc.set_clim(cmin, cmax)

    if add_colorbar:
        plt.colorbar(pc, ax=ax, label=label, orientation="vertical")
    pc.set_array(values)

    if "edge_color" in kwargs:
        edge_color = kwargs["edge_color"]
    else:
        edge_color = None
    pc.set_edgecolor(edge_color)

    ax.add_collection(pc)
    ax.autoscale()
    ax.set_xlabel("relative distance [m]")
    ax.set_ylabel("z [m]")

    if title is not None:
        ax.set_title(title)

    return ax


def _Get_2DVertical_elements(element_table: np.ndarray) -> np.ndarray:
    # if (type == DfsuFileType.DfsuVerticalProfileSigmaZ) or (
    #     type == DfsuFileType.DfsuVerticalProfileSigma
    # ):
    elements = [list(element_table[i]) for i in range(len(list(element_table)))]
    return np.asarray(elements)
