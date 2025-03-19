import marimo

__generated_with = "0.11.21"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        r"""
        # Flexible Mesh - Plotting

        """
    )
    return


@app.cell
def _():
    import mikeio
    import matplotlib.pyplot as plt
    return mikeio, plt


@app.cell
def _(mikeio):
    dfs = mikeio.open('../tests/testdata/FakeLake.dfsu')
    dfs
    return (dfs,)


@app.cell
def _(dfs):
    geom = dfs.geometry
    return (geom,)


@app.cell
def _(geom):
    geom.plot()
    return


@app.cell
def _(geom):
    geom.plot.mesh()
    return


@app.cell
def _(geom):
    geom.plot(vmin=-30)
    return


@app.cell
def _(geom):
    geom.plot.contour(show_mesh=True, levels=16, cmap='tab20', vmin=-30)
    return


@app.cell
def _(geom):
    geom.plot(plot_type='contourf', show_mesh=True, levels=6, vmin=-30)
    return


@app.cell
def _(geom):
    geom.plot(plot_type='shaded', show_mesh=False, vmin=-30)
    return


@app.cell
def _(geom):
    geom.plot(plot_type='shaded', add_colorbar=False)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
