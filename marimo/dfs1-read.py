import marimo

__generated_with = "0.10.2"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        r"""
        # Dfs1 - Read
        """
    )
    return


@app.cell
def _():
    import mikeio
    return (mikeio,)


@app.cell
def _(mikeio):
    ds = mikeio.read("../tests/testdata/random.dfs1")
    ds
    return (ds,)


@app.cell
def _(ds):
    ds.geometry
    return


@app.cell
def _(ds):
    da = ds["testing water level"]
    return (da,)


@app.cell
def _(da):
    da.plot();
    return


@app.cell
def _(da):
    da.isel(x=0).plot(title="First");
    return


@app.cell
def _(da):
    da.isel(x=-1).plot(title="Last");
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Export a single point from the Dfs1 to Dfs0
        """
    )
    return


@app.cell
def _(da):
    da.isel(x=0).to_dfs("random_0.dfs0")
    return


@app.cell
def _(mikeio):
    mikeio.read("random_0.dfs0")[0].plot();
    return


@app.cell
def _():
    import os
    os.remove("random_0.dfs0")
    return (os,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

