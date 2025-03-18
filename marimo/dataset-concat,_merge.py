import marimo

__generated_with = "0.10.2"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        r"""
        # Combining Datasets

        Datasets can be combined along the items and time axis
        """
    )
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import mikeio
    return mikeio, plt


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Concatenate Datasets (along the time axis)
        """
    )
    return


@app.cell
def _(mikeio):
    ds1 = mikeio.read("../tests/testdata/tide1.dfs1")
    ds1
    return (ds1,)


@app.cell
def _(mikeio):
    ds2 = mikeio.read("../tests/testdata/tide2.dfs1") + 0.5  # add offset
    ds2
    return (ds2,)


@app.cell
def _(mo):
    mo.md(
        r"""
        Concatenating data along the time axis can be done with `Dataset.concat`
        """
    )
    return


@app.cell
def _(ds1, ds2, mikeio):
    ds3 = mikeio.Dataset.concat([ds1, ds2])
    ds3
    return (ds3,)


@app.cell
def _(ds1, ds2, ds3, plt):
    plt.plot(ds1.time, ds1[0].to_numpy()[:,1], 'ro', label="Dataset 1")
    plt.plot(ds2.time, ds2[0].to_numpy()[:,1], 'k+', label="Dataset 2")
    plt.plot(ds3.time, ds3[0].to_numpy()[:,1], 'g-', label="Dataset 3")
    plt.title("Notice the offset...")
    plt.legend();
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Merging datasets
        """
    )
    return


@app.cell
def _(mikeio):
    dsA = mikeio.read("../tests/testdata/tide1.dfs1")
    dsA
    return (dsA,)


@app.cell
def _(dsA):
    dsB = dsA.copy()
    dsB = dsB.rename({"Level":"Other_level"})
    dsB = dsB + 2
    dsB
    return (dsB,)


@app.cell
def _(mo):
    mo.md(
        r"""
        Merge datasets with different items can be done like this:
        """
    )
    return


@app.cell
def _(dsA, dsB, mikeio):
    dsC = mikeio.Dataset.merge([dsA, dsB])
    dsC
    return (dsC,)


@app.cell
def _(mo):
    mo.md(
        r"""
        Which in this simple case with a single item in each dataset is equivalent to:
        """
    )
    return


@app.cell
def _(dsA):
    daA = dsA[0]
    daA
    return (daA,)


@app.cell
def _(dsB):
    daB = dsB[0]
    daB
    return (daB,)


@app.cell
def _(daA, daB, mikeio):
    mikeio.Dataset([daA, daB])
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

