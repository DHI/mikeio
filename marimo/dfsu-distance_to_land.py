import marimo

__generated_with = "0.11.21"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        r"""
        # Dfsu - Distance to land
        Calculate the distance to land for each element in mesh and save to dfsu file
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import mikeio
    return mikeio, np, plt


@app.cell
def _(mikeio):
    msh = mikeio.Mesh("../tests/testdata/odense_rough.mesh")
    msh
    return (msh,)


@app.cell
def _(msh):
    msh.plot()
    return


@app.cell
def _(msh):
    msh.geometry.plot.outline()
    return


@app.cell
def _(mo):
    mo.md(r"""## Get a list of land nodes""")
    return


@app.cell
def _(msh, plt):
    ncland = msh.node_coordinates[msh.codes==1]

    plt.scatter(ncland[:,0], ncland[:,1])
    return (ncland,)


@app.cell
def _(mo):
    mo.md(r"""## Get element coordinates""")
    return


@app.cell
def _(msh):
    ec = msh.element_coordinates
    return (ec,)


@app.cell
def _(mo):
    mo.md(r"""## Calculate distance to nearest land node""")
    return


@app.cell
def _(ec, ncland, np):
    d = np.min(np.sqrt((ec[:, np.newaxis, 0] - ncland[:, 0])**2 + 
                        (ec[:, np.newaxis, 1] - ncland[:, 1])**2), axis=1)
    return (d,)


@app.cell
def _(mo):
    name = mo.ui.text(placeholder="Distance to land")
    name
    return (name,)


@app.cell
def _(d, mikeio, msh, name):
    da = mikeio.DataArray(data=d,
                          geometry=msh.geometry,
                          item=mikeio.ItemInfo(name.value, mikeio.EUMType.Distance, mikeio.EUMUnit.meter))
    da
    return (da,)


@app.cell
def _(mo):
    threshold = mo.ui.slider(500,2000,100)
    threshold
    return (threshold,)


@app.cell
def _(da, threshold):
    da.plot(title="", levels=[0,threshold.value])
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
