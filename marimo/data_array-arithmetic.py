import marimo

__generated_with = "0.10.2"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        r"""
        # DataArray - Arithmetic

        We can basic arithmetic operations (plus, minus, multiplication and division) with DataArrays, both with scalars, numpy arrays and other DataArrays. The output will in all cases be a new DataArray.
        """
    )
    return


@app.cell
def _():
    import mikeio
    return (mikeio,)


@app.cell
def _(mikeio):
    fn1 = "../tests/testdata/oresundHD_run1.dfsu"
    da1 = mikeio.read(fn1, items="Surface elevation")[0]
    fn2 = "../tests/testdata/oresundHD_run2.dfsu"
    da2 = mikeio.read(fn2, items="Surface elevation")[0]
    return da1, da2, fn1, fn2


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Multiply or add scalar

        We can scale a DataArray or add a constant using *, +, / and -
        """
    )
    return


@app.cell
def _(da1):
    da1.values.mean()
    return


@app.cell
def _(da1):
    da1_A = da1 + 1
    da1_B = da1 - 1
    da1_A.values.mean(), da1_B.values.mean()
    return da1_A, da1_B


@app.cell
def _(da1):
    da1_C = da1 * 2
    da1_D = da1 / 2
    da1_C.values.mean(), da1_D.values.mean()
    return da1_C, da1_D


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Difference between two DataArrays

        Assume that we have two calibration runs and we wan't to find the difference...
        """
    )
    return


@app.cell
def _(da1, da2):
    _da_diff = da1 - da2
    _da_diff.plot(title='Difference')
    return


@app.cell
def _(da1, da2):
    _da_diff = da1 / da2
    _da_diff.plot(title='da1/da2')
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
