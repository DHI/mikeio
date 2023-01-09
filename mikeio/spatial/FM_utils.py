import warnings

import numpy as np

from .utils import _relative_cumulative_distance


def _plot_map(
    node_coordinates,
    element_table,
    element_coordinates,
    boundary_polylines,
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
    boundary_polylines,
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

    import matplotlib.cm as cm
    import matplotlib.colors as mplc
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    mesh_col = "0.95"
    mesh_col_dark = "0.6"

    if plot_type is None:
        plot_type = "outline_only"

    plot_data = True
    if plot_type == "mesh_only" or plot_type == "outline_only":
        plot_data = False

    if plot_type == "mesh_only":
        show_mesh = True

    if cmap is None:
        cmap = cm.viridis

    nc = node_coordinates
    ec = element_coordinates
    ne = ec.shape[0]

    is_bathy = False
    if z is None:
        is_bathy = True
        if plot_data:
            z = ec[:, 2]
            if label is None:
                label = "Bathymetry (m)"
    else:
        # if isinstance(z, DataArray):
        #     z = z.to_numpy().copy()
        # if isinstance(z, Dataset) and len(z) == 1:  # if single-item Dataset
        #     z = z[0].to_numpy().copy()
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

    if ((vmin is not None) or (vmax is not None)) and (
        levels is not None and not np.isscalar(levels)
    ):
        raise ValueError(
            "vmin/vmax cannot be provided together with non-integer levels"
        )

    if elements is not None:
        if plot_type.startswith("contour"):
            raise ValueError("elements argument not possible with contour plots")
        newz = np.full_like(z, fill_value=np.nan)
        newz[elements] = z[elements]
        z = newz

    if plot_data and vmin is None:
        vmin = np.nanmin(z)
    if plot_data and vmax is None:
        vmax = np.nanmax(z)

    if plot_data and vmin == vmax:
        vmin = vmin - 0.1
        vmax = vmin + 0.2

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

        if isinstance(cmap, str):
            cmap = cm.get_cmap(cmap)
        cmap_norm = mplc.BoundaryNorm(levels, cmap.N)
        cmap_ScMappable = cm.ScalarMappable(cmap=cmap, norm=cmap_norm)
    if ("contour" in plot_type) and (levels is None):
        n_levels = 10
        levels = np.linspace(vmin, vmax, n_levels)

    cbar_extend = _cbar_extend(z, vmin, vmax)

    # plot in existing or new axes?
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # set aspect ratio
    is_geo = projection == "LONG/LAT"
    if is_geo:
        mean_lat = np.mean(nc[:, 1])
        ax.set_aspect(1.0 / np.cos(np.pi * mean_lat / 180))
    else:
        ax.set_aspect("equal")

    _set_xy_label_by_projection(ax, projection)

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
        patches = _to_polygons(nc, element_table)
        fig_obj = PatchCollection(
            patches, edgecolor=mesh_col_dark, facecolor="none", linewidths=0.3
        )
        ax.add_collection(fig_obj)

    elif plot_type == "patch" or plot_type == "box":
        patches = _to_polygons(nc, element_table)
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
        if show_mesh and _is_tri_only(element_table):
            mesh_linewidth = 0.4
            if n_refinements > 0:
                n_refinements = 0
                print("Warning: mesh refinement is not possible if plot_mesh=True")

        elem_table, ec, z = _create_tri_only_element_table(
            nc, element_table, ec, data=z
        )
        triang = tri.Triangulation(nc[:, 0], nc[:, 1], elem_table)

        zn = _get_node_centered_data(nc, elem_table, ec, z)

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
                cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
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
            # vbuf = 0.01 * (vmax - vmin) / n_levels
            # avoid white outside limits
            # zn = np.clip(zn, vmin + vbuf, vmax - vbuf) # # THIS LINE SEEMS TO CAUSE TROUBLE
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

            # colorbar
            if add_colorbar:
                cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
                if cmap_ScMappable is None:
                    plt.colorbar(fig_obj, label=label, cax=cax)
                else:
                    try:
                        plt.colorbar(
                            cmap_ScMappable,
                            label=label,
                            cax=cax,
                            ticks=levels,
                            extend=cbar_extend,
                        )
                    except:
                        warnings.warn("Cannot add colorbar")

        else:
            if (plot_type is not None) and plot_type != "outline_only":
                raise Exception(f"plot_type {plot_type} unknown!")

        if show_mesh and (not _is_tri_only(element_table)):
            # if mesh is not tri only, we need to add it manually on top
            patches = _to_polygons(nc, element_table)
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
        for exterior in boundary_polylines.exteriors:
            ax.plot(*exterior.xy.T, color=out_col, linewidth=linwid)
            xd, yd = exterior.xy[:, 0], exterior.xy[:, 1]
            xmin, xmax = min(xmin, np.min(xd)), max(xmax, np.max(xd))
            ymin, ymax = min(ymin, np.min(yd)), max(ymax, np.max(yd))

        for interior in boundary_polylines.interiors:
            ax.plot(*interior.xy.T, color=out_col, linewidth=linwid)

    # set plot limits
    xybuf = 6e-3 * (xmax - xmin)
    ax.set_xlim(xmin - xybuf, xmax + xybuf)
    ax.set_ylim(ymin - xybuf, ymax + xybuf)

    if title is not None:
        ax.set_title(title)

    return ax


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


def _is_tri_only(element_table):
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
    elem_table, ec, data = _create_tri_only_element_table(
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


def _create_tri_only_element_table(
    node_coordinates, element_table, element_coordinates, data=None
):
    """Convert quad/tri mesh to pure tri-mesh"""

    if _is_tri_only(element_table):
        # already tri-only? just convert to 2d array
        return np.stack(element_table), element_coordinates, data

    ec = element_coordinates.copy()

    if data is None:
        data = []

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


def _cbar_extend(calc_data, vmin, vmax):
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


def _point_in_polygon(xn: np.array, yn: np.array, xp: float, yp: float) -> bool:
    """Check for each side in the polygon that the point is on the correct side"""

    for j in range(len(xn) - 1):
        if (yn[j + 1] - yn[j]) * (xp - xn[j]) + (-xn[j + 1] + xn[j]) * (yp - yn[j]) > 0:
            return False
    if (yn[0] - yn[-1]) * (xp - xn[-1]) + (-xn[0] + xn[-1]) * (yp - yn[-1]) > 0:
        return False
    return True


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
