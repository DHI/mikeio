import marimo

__generated_with = "0.10.2"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        r"""
        # Dataset - Plotting

        For most plotting purposes the DataArray rather than the Dataset are used (see other other notebooks on how-to). 

        But for comparison of different items, the Dataset.plot method can be useful. 
        """
    )
    return


@app.cell
def _():
    import mikeio
    return (mikeio,)


@app.cell
def _(mikeio):
    ds = mikeio.read("../tests/testdata/wind_north_sea.dfsu")
    ds
    return (ds,)


@app.cell
def _(ds):
    ds.plot.scatter(x="Wind speed", y="Wind direction");
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

