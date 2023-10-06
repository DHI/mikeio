
import numpy as np
from collections import namedtuple

from ._utils import _relative_cumulative_distance


MESH_COL = "0.95"
MESH_COL_DARK = "0.6"

BoundaryPolylines = namedtuple(
    "BoundaryPolylines",
    ["n_exteriors", "exteriors", "n_interiors", "interiors"],
)


def _plot_map(
    node_coordinates,
    element_table,
    element_coordinates,
    boundary_polylines: BoundaryPolylines,
    projection="",
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
    show_mesh=False,
    show_outline=True,
    figsize=None,
    ax=None,
    add_colorbar=True,
):
    """
    Plot unstructured data and/or mesh, mesh outline

    Parameters
    ----------
    node_coordinates,
    element_table,
    element_coordinates,
    boundary_polylines: BoundaryPolylines,
    projection,
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

    if elements is not None:
        if plot_type.startswith("contour"):
            raise ValueError("elements argument not possible with contour plots")
        newz = np.full_like(z, fill_value=np.nan)
        newz[elements] = z[elements]
        z = newz

    assert len(z) == ec.shape[0]

    label = label or ""

    vmin, vmax, cmap, cmap_norm, cmap_ScMappable, levels = __set_colormap_levels(
        cmap, vmin, vmax, levels, z
    )
    cbar_extend = __cbar_extend(z, vmin, vmax)

    if plot_type == "patch":
        fig_obj = __plot_patch(
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

    ax.set_title(title)

    return ax


def __set_colormap_levels(cmap, vmin, vmax, levels, z):
    """Set colormap, levels, vmin, vmax, and cmap_norm

    Parameters
    ----------
    cmap : str or matplotlib.colors.Colormap
        colormap name or colormap object
    vmin : float
        minimum value for colorbar
    vmax : float
        maximum value for colorbar
    levels : int or list of float
        number of levels or list of levels
    z : array of float
        data to be plotted

    Returns
    -------
    vmin : float
        minimum value for colorbar
    vmax : float
        maximum value for colorbar
    cmap : matplotlib.colors.Colormap
        colormap object
    cmap_norm : matplotlib.colors.Normalize
        colormap normalization object
    cmap_ScMappable : matplotlib.cm.ScalarMappable
        colormap object
    levels : list of float
        list of levels
    """

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
            levels = np.linspace(vmin, vmax, n_levels)
        else:
            n_levels = len(levels)
            vmin = min(levels)
            vmax = max(levels)

        levels = np.array(levels)

        if isinstance(cmap, str):
            cmap = matplotlib.colormaps[cmap]
        cmap_norm = mplc.BoundaryNorm(levels, cmap.N)
        cmap_ScMappable = cm.ScalarMappable(cmap=cmap, norm=cmap_norm)

    if levels is None:
        levels = np.linspace(vmin, vmax, 10)

    return vmin, vmax, cmap, cmap_norm, cmap_ScMappable, levels


def __set_plot_limits(ax, nc) -> None:
    """Set default plot limits

    Override with matplotlib ax.set_xlim, ax.set_ylim
    """
    xmin, xmax = nc[:, 0].min(), nc[:, 0].max()
    ymin, ymax = nc[:, 1].min(), nc[:, 1].max()

    xybuf = 6e-3 * (xmax - xmin)
    ax.set_xlim(xmin - xybuf, xmax + xybuf)
    ax.set_ylim(ymin - xybuf, ymax + xybuf)


def __plot_mesh_only(ax, nc, element_table):
    """plot mesh only (no data)

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        axes object
    nc : array of float
        node coordinates
    element_table : array of int
        element table

    Returns
    -------
    matplotlib.axes.Axes
        axes object
    """
    from matplotlib.collections import PatchCollection

    patches = _to_polygons(nc, element_table)
    fig_obj = PatchCollection(
        patches, edgecolor=MESH_COL_DARK, facecolor="none", linewidths=0.3
    )
    ax.add_collection(fig_obj)


def __plot_outline_only(ax, boundary_polylines: BoundaryPolylines):
    """plot outline only (no data)

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        axes object
    boundary_polylines : BoundaryPolylines
        boundary polylines

    Returns
    -------
    matplotlib.axes.Axes
        axes object
    """
    __add_outline(ax, boundary_polylines)
    return ax


def __plot_patch(ax, nc, element_table, show_mesh, cmap, cmap_norm, z, vmin, vmax):
    """plot patch with data from z

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        axes object
    nc : array of float
        node coordinates
    element_table : array of int
        element table
    show_mesh : bool
        include mesh polygons
    cmap : str or matplotlib.colors.Colormap
        colormap name or colormap object
    cmap_norm : matplotlib.colors.Normalize
        colormap normalization object
    z : array of float
        data to be plotted
    vmin : float
        minimum value for colorbar
    vmax : float
        maximum value for colorbar

    Returns
    -------
    matplotlib.axes.Axes
        axes object
    """

    from matplotlib.collections import PatchCollection

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


def __get_tris(nc, element_table, ec, z, n_refinements):
    """get triangulation object and node-centered data

    Parameters
    ----------
    nc : array of float
        node coordinates
    element_table : array of int
        element table
    ec : array of int
        element coordinates
    z : array of float
        data to be plotted
    n_refinements : int
        number of refinements

    Returns
    -------
    matplotlib.tri.Triangulation and node-centered data
    """

    import matplotlib.tri as tri

    elem_table, ec, z = __create_tri_only_element_table(nc, element_table, ec, data=z)
    triang = tri.Triangulation(nc[:, 0], nc[:, 1], elem_table)

    zn = _get_node_centered_data(nc, elem_table, ec, z)

    if n_refinements > 0:
        # TODO: refinements doesn't seem to work for 3d files?
        refiner = tri.UniformTriRefiner(triang)
        triang, zn = refiner.refine_field(zn, subdiv=n_refinements)

    return triang, zn


def __add_colorbar(ax, cmap_ScMappable, fig_obj, label, levels, cbar_extend) -> None:
    """add colorbar to axes

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        axes object
    cmap_ScMappable : matplotlib.cm.ScalarMappable
        colormap object
    fig_obj : matplotlib.figure.Figure
        figure object
    label : str
        colorbar label
    levels : array of float
        colorbar levels
    cbar_extend : str
        extend colorbar beyond min/max values

    Returns
    -------
    None
    """

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt

    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    cmap_sm = cmap_ScMappable if cmap_ScMappable else fig_obj

    plt.colorbar(
        cmap_sm,
        label=label,
        cax=cax,
        ticks=levels,
        boundaries=levels,
        extend=cbar_extend,
    )


def __set_aspect_ratio(ax, nc, projection):
    """set aspect ratio

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        axes object
    nc : array of float
        node coordinates
    projection : str
        projection type

    Returns
    -------
    None
    """
    is_geo = projection == "LONG/LAT"
    if is_geo:
        mean_lat = np.mean(nc[:, 1])
        ax.set_aspect(1.0 / np.cos(np.pi * mean_lat / 180))
    else:
        ax.set_aspect("equal")


def __add_non_tri_mesh(ax, nc, element_table, plot_type) -> None:
    """add non-triangular mesh to axes

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        axes object
    nc : array of float
        node coordinates
    element_table : array of int
        element table
    plot_type : str

    Returns
    -------
    None
    """
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


def __add_outline(ax, boundary_polylines: BoundaryPolylines) -> None:
    """add outline to axes

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        axes object
    boundary_polylines: BoundaryPolylines
        boundary polylines

    Returns
    -------
    None
    """

    lines = boundary_polylines.exteriors + boundary_polylines.interiors
    for line in lines:
        ax.plot(*line.xy.T, color="0.4", linewidth=1.2)


def _set_xy_label_by_projection(ax, projection):
    if (not projection) or projection == "NON-UTM":
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
    elif projection == "LONG/LAT":
        ax.set_xlabel("Longitude [degrees]")
        ax.set_ylabel("Latitude [degrees]")
    else:
        ax.set_xlabel("Easting [m]")
        ax.set_ylabel("Northing [m]")


def __is_tri_only(element_table):
    return max([len(el) for el in element_table]) == 3


def _to_polygons(node_coordinates, element_table):
    """generate matplotlib polygons from element table for plotting

    Returns
    -------
    list(matplotlib.patches.Polygon)
        list of polygons for plotting
    """
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


def _get_node_centered_data(
    node_coordinates, element_table, element_coordinates, data, extrapolate=True
):
    """convert cell-centered data to node-centered by pseudo-laplacian method

    Parameters
    ----------
    node_coordinates,
    element_table,
    element_coordinates
    data : np.array(float)
        cell-centered data
    extrapolate : bool, optional
        allow the method to extrapolate, default:True

    Returns
    -------
    np.array(float)
        node-centered data
    """
    nc = node_coordinates
    elem_table, ec, data = __create_tri_only_element_table(
        nc, element_table, element_coordinates, data
    )

    node_cellID = [
        list(np.argwhere(elem_table == i)[:, 0])
        for i in np.unique(
            elem_table.reshape(
                -1,
            )
        )
    ]
    node_centered_data = np.zeros(shape=nc.shape[0])
    for n, item in enumerate(node_cellID):
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
    node_coordinates, element_table, element_coordinates, data
):
    """Convert quad/tri mesh to pure tri-mesh"""

    if __is_tri_only(element_table):
        # already tri-only? just convert to 2d array
        return np.stack(element_table), element_coordinates, data

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


def __cbar_extend(calc_data, vmin, vmax) -> str:
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
    node_coordinates,
    element_table,
    values,
    zn=None,
    is_geo=False,
    cmin=None,
    cmax=None,
    label="",
    add_colorbar=True,
    **kwargs,
):
    """
    Plot unstructured vertical profile

    Parameters
    ----------
    node_coordinates: np.array
    element_table: np.array[np.array]
    values: np.array
        value for each element to plot
    zn: np.array, optional
        dynamic vertical node positions,
        default: use static vertical positions
    is_geo: bool, optional
        are coordinates geographical (for calculating
        relative distance in meters), default: False
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
    edge_color: str, optional
        color of mesh lines, default: None
    add_colorbar: bool, optional
        Add colorbar to plot, default True
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

    nc = node_coordinates
    s_coordinate = _relative_cumulative_distance(nc, is_geo=is_geo)
    z_coordinate = nc[:, 2] if zn is None else zn

    elements = _Get_2DVertical_elements(element_table)

    # plot in existing or new axes?
    if "ax" in kwargs:
        ax = kwargs["ax"]
    else:
        figsize = None
        if "figsize" in kwargs:
            figsize = kwargs["figsize"]
        _, ax = plt.subplots(figsize=figsize)

    sz = np.c_[s_coordinate, z_coordinate]
    verts = sz[elements]

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

    if "title" in kwargs:
        ax.set_title(kwargs["title"])

    return ax


def _Get_2DVertical_elements(element_table):
    # if (type == DfsuFileType.DfsuVerticalProfileSigmaZ) or (
    #     type == DfsuFileType.DfsuVerticalProfileSigma
    # ):
    elements = [list(element_table[i]) for i in range(len(list(element_table)))]
    return np.asarray(elements)
