import numpy as np
import warnings


class UnstructuredPlotter:
    def __init__(self, geometry):

        self._geometry = geometry

    def _to_polygons(self, geometry=None):
        """generate matplotlib polygons from element table for plotting

        Returns
        -------
        list(matplotlib.patches.Polygon)
            list of polygons for plotting
        """
        if geometry is None:
            geometry = self._geometry
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

    def _create_tri_only_element_table(self, data=None, geometry=None):
        """Convert quad/tri mesh to pure tri-mesh
        """
        if geometry is None:
            geometry = self

        ec = geometry.element_coordinates
        if geometry.is_tri_only:
            return np.asarray(geometry.element_table), ec, data

        if data is None:
            data = []

        elem_table = [
            list(geometry.element_table[i]) for i in range(geometry.n_elements)
        ]
        tmp_elmnt_nodes = elem_table.copy()
        for el, item in enumerate(tmp_elmnt_nodes):
            if len(item) == 4:
                elem_table.pop(el)  # remove quad element

                # insert two new tri elements in table
                elem_table.insert(el, item[:3])
                tri2_nodes = [item[i] for i in [2, 3, 0]]
                elem_table.append(tri2_nodes)

                # new center coordinates for new tri-elements
                ec[el] = geometry.node_coordinates[item[:3]].mean(axis=1)
                tri2_ec = geometry.node_coordinates[tri2_nodes].mean(axis=1)
                ec = np.append(ec, tri2_ec.reshape(1, -1), axis=0)

                # use same data in two new tri elements
                data = np.append(data, data[el])

        return np.asarray(elem_table), ec, data

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
        levels=10,
        n_refinements=0,
        show_mesh=True,
        show_outline=True,
        figsize=None,
        ax=None,
    ):
        """
        Plot unstructured data and/or mesh, mesh outline  

        Parameters
        ----------
        z: np.array, optional
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

        Returns
        -------
        <matplotlib.axes>          
        """

        import matplotlib.cm as cm
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
            if self._geometry.is_2d:
                geometry = self._geometry
            else:
                geometry = self._geometry.geometry2d
        else:
            # spatial subset
            if self._geometry.is_2d:
                geometry = self._geometry.elements_to_geometry(elements)
            else:
                geometry = self._geometry.elements_to_geometry(
                    elements, node_layers="bottom"
                )

        nc = geometry.node_coordinates
        ec = geometry.element_coordinates
        ne = ec.shape[0]

        if z is None:
            if plot_data:
                z = ec[:, 2]
                if label is None:
                    label = "Bathymetry (m)"
        else:
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
        if "contour" in plot_type:
            if levels is None:
                levels = 10
            if np.isscalar(levels):
                n_levels = levels
                levels = np.linspace(vmin, vmax, n_levels)
            else:
                n_levels = len(levels)
                vmin = min(levels)
                vmax = max(levels)

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
            patches = self._to_polygons()
            fig_obj = PatchCollection(
                patches, edgecolor=mesh_col_dark, facecolor="none", linewidths=0.3
            )
            ax.add_collection(fig_obj)

        elif plot_type == "patch" or plot_type == "box":
            patches = self._to_polygons()
            # do plot as patches (like MZ "box contour")
            # with (constant) element center values
            if show_mesh:
                fig_obj = PatchCollection(
                    patches, cmap=cmap, edgecolor=mesh_col, linewidths=0.4
                )
            else:
                fig_obj = PatchCollection(
                    patches, cmap=cmap, edgecolor="face", alpha=None, linewidths=None
                )

            fig_obj.set_array(z)
            fig_obj.set_clim(vmin, vmax)
            ax.add_collection(fig_obj)

            cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
            plt.colorbar(fig_obj, label=label, cax=cax)

        else:
            # do node-based triangular plot
            import matplotlib.tri as tri

            mesh_linewidth = 0.0
            if show_mesh and geometry.is_tri_only:
                mesh_linewidth = 0.4
                if n_refinements > 0:
                    n_refinements = 0
                    print("Warning: mesh refinement is not possible if plot_mesh=True")

            elem_table, ec, z = self._geometry._create_tri_only_element_table(
                data=z, geometry=geometry
            )
            triang = tri.Triangulation(nc[:, 0], nc[:, 1], elem_table)

            zn = geometry.get_node_centered_data(z)

            if n_refinements > 0:
                # TODO: refinements doesn't seem to work for 3d files?
                refiner = tri.UniformTriRefiner(triang)
                triang, zn = refiner.refine_field(zn, subdiv=n_refinements)

            if plot_type == "shaded" or plot_type == "smooth":
                ax.triplot(triang, lw=mesh_linewidth, color=mesh_col)
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

                cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
                plt.colorbar(fig_obj, label=label, cax=cax)

            elif plot_type == "contour" or plot_type == "contour_lines":
                ax.triplot(triang, lw=mesh_linewidth, color=mesh_col_dark)
                fig_obj = ax.tricontour(
                    triang, zn, levels=levels, linewidths=[1.2], cmap=cmap
                )
                ax.clabel(fig_obj, fmt="%1.2f", inline=1, fontsize=9)
                if len(label) > 0:
                    ax.set_title(label)

            elif plot_type == "contourf" or plot_type == "contour_filled":
                ax.triplot(triang, lw=mesh_linewidth, color=mesh_col)
                vbuf = 0.01 * (vmax - vmin) / n_levels
                zn = np.clip(zn, vmin + vbuf, vmax - vbuf)  # avoid white outside limits
                fig_obj = ax.tricontourf(triang, zn, levels=levels, cmap=cmap)

                # colorbar
                cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
                plt.colorbar(fig_obj, label=label, cax=cax)

            else:
                if (plot_type is not None) and plot_type != "outline_only":
                    raise Exception(f"plot_type {plot_type} unknown!")

            if show_mesh and (not geometry.is_tri_only):
                # if mesh is not tri only, we need to add it manually on top
                patches = self._to_polygons()
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
            try:
                domain = self._shapely_domain2d
            except:
                warnings.warn("Could not plot outline. Failed to convert to_shapely()")
            try:
                if domain:
                    out_col = "0.4"
                    ax.plot(*domain.exterior.xy, color=out_col, linewidth=1.2)
                    xd, yd = domain.exterior.xy[0], domain.exterior.xy[1]
                    xmin, xmax = min(xmin, np.min(xd)), max(xmax, np.max(xd))
                    ymin, ymax = min(ymin, np.min(yd)), max(ymax, np.max(yd))
                    for j in range(len(domain.interiors)):
                        interj = domain.interiors[j]
                        ax.plot(*interj.xy, color=out_col, linewidth=1.2)
            except:
                warnings.warn("Could not plot outline")

        # set plot limits
        xybuf = 6e-3 * (xmax - xmin)
        ax.set_xlim(xmin - xybuf, xmax + xybuf)
        ax.set_ylim(ymin - xybuf, ymax + xybuf)

        if title is not None:
            ax.set_title(title)

        return ax
