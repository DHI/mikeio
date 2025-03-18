import marimo

__generated_with = "0.10.2"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        r"""
        # Dfsu and Mesh - Plotting
        Demonstrate different ways of plotting dfsu and mesh files. This includes plotting

        * outline_only
        * mesh_only
        * patch - similar to MIKE Zero box contour)
        * contour - contour lines
        * contourf - filled contours
        * shaded
        """
    )
    return


@app.cell
def _():
    import mikeio
    import matplotlib.pyplot as plt
    from matplotlib_inline.backend_inline import set_matplotlib_formats
    set_matplotlib_formats('png')
    plt.rcParams["figure.figsize"] = (10,8)
    return mikeio, plt, set_matplotlib_formats


@app.cell
def _(mo):
    mo.md(
        r"""
        # Load dfsu file
        """
    )
    return


@app.cell
def _(filename, mikeio):
    _filename = '../tests/testdata/FakeLake.dfsu'
    dfs = mikeio.open(filename)
    dfs
    return (dfs,)


@app.cell
def _(dfs):
    geom = dfs.geometry
    geom
    return (geom,)


@app.cell
def _(geom):
    geom.plot();
    return


@app.cell
def _(geom):
    geom.plot.mesh();
    return


@app.cell
def _(geom):
    geom.plot(vmin=-30);
    return


@app.cell
def _(geom):
    geom.plot.contour(show_mesh=True, levels=16, cmap='tab20', vmin=-30);
    return


@app.cell
def _(geom):
    geom.plot(plot_type='contourf', show_mesh=True, levels=6, vmin=-30);
    return


@app.cell
def _(geom):
    geom.plot(plot_type='shaded', show_mesh=False, vmin=-30);
    return


@app.cell
def _(geom):
    geom.plot(plot_type='shaded', add_colorbar=False);
    return


@app.cell
def _(geom):
    geom.isel(range(400,600)).plot(plot_type='patch', vmin=-30, figsize=(4,6));
    return


@app.cell
def _(geom, plt):
    fig, ax = plt.subplots(3, 2)
    geom.plot(title='patch', ax=_ax[0, 0])
    geom.plot.contourf(title='contourf', levels=5, ax=_ax[0, 1])
    geom.plot(plot_type='shaded', title='shaded', ax=_ax[1, 0])
    geom.plot.contour(title='contour', show_mesh=True, levels=6, vmin=-30, ax=_ax[1, 1])
    geom.plot.mesh(title='mesh_only', ax=_ax[2, 0])
    geom.plot.outline(title='outline_only', ax=_ax[2, 1])
    plt.tight_layout()
    return ax, fig


@app.cell
def _(mo):
    mo.md(
        r"""
        # Plot data from surface layer of 3d dfsu file
        """
    )
    return


@app.cell
def _(filename, mikeio):
    _filename = '../tests/testdata/oresund_sigma_z.dfsu'
    dfs_1 = mikeio.open(filename)
    dfs_1
    return (dfs_1,)


@app.cell
def _(dfs_1):
    da = dfs_1.read(items='Salinity', layers='top', time=0)[0]
    da
    return (da,)


@app.cell
def _(da):
    da.plot(cmap='plasma');
    return


@app.cell
def _(da):
    da.plot(add_colorbar=False);
    return


@app.cell
def _(da):
    _ax = da.plot.contour(show_mesh=True, cmap='tab20', levels=[11, 13, 15, 17, 18, 19, 20, 20.5])
    _ax.set_ylim(6135000, 6160000)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # plot data from a z-layer
        """
    )
    return


@app.cell
def _(dfs_1):
    da_1 = dfs_1.read(items='Salinity', layers=3, time=0)[0]
    da_1
    return (da_1,)


@app.cell
def _(da_1, dfs_1):
    _ax = da_1.plot(cmap='plasma')
    dfs_1.geometry.plot.outline(ax=_ax, title=None)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

