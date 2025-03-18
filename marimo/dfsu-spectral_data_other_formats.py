import marimo

__generated_with = "0.10.2"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        r"""
        # More Dfsu spectral files
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
        ## Output from directional-sector-MIKE 21 SW run

        MIKE 21 SW can be run with dicretized directions only in a directional sector. The reading and plotting of such spectra are also supported in MIKE IO.
        """
    )
    return


@app.cell
def _(mikeio):
    fn = "../tests/testdata/spectra/MIKE21SW_dir_sector_area_spectra.dfsu"
    dfs = mikeio.open(fn)
    dfs
    return dfs, fn


@app.cell
def _(dfs):
    dfs.geometry.is_spectral
    return


@app.cell
def _(dfs):
    da = dfs.read(time=0)["Energy density"]
    da
    return (da,)


@app.cell
def _(da):
    da.plot();
    return


@app.cell
def _(da):
    da_pt = da.isel(element=0)
    da_pt
    return (da_pt,)


@app.cell
def _(da_pt):
    da_pt.plot(rmax=10, vmin=0);
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Frequency spectra

        Frequency spectra have 0 directions. They can be of type point, line and area.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Point frequency spectrum
        """
    )
    return


@app.cell
def _():
    fn_1 = '../tests/testdata/spectra/pt_freq_spectra.dfsu'
    return (fn_1,)


@app.cell
def _(fn_1, mikeio):
    da_1 = mikeio.read(fn_1)[0]
    da_1
    return (da_1,)


@app.cell
def _(da_1):
    da_1.sel(time='2017-10-27 02:00').plot()
    return


@app.cell
def _(da_1):
    da_1.frequencies
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Compute significant wave height time series
        """
    )
    return


@app.cell
def _(da_1):
    Hm0 = da_1.to_Hm0()
    Hm0
    return (Hm0,)


@app.cell
def _(Hm0):
    Hm0.plot();
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Area frequency spectra
        """
    )
    return


@app.cell
def _(mikeio):
    fn_2 = '../tests/testdata/spectra/area_freq_spectra.dfsu'
    da_2 = mikeio.read(fn_2)[0]
    da_2
    return da_2, fn_2


@app.cell
def _(da_2):
    (da_2.n_frequencies, da_2.n_directions)
    return


@app.cell
def _(da_2):
    da_2.plot()
    return


@app.cell
def _(da_2):
    da_2.sel(x=2.7, y=52.4).plot()
    return


@app.cell
def _(da_2, plt):
    elem = 0
    plt.plot(da_2.frequencies, da_2[:, elem].to_numpy().T)
    plt.legend(da_2.time)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('directionally integrated energy [m*m*s]')
    plt.title(f'Area dfsu file, frequency spectrum in element {elem}')
    return (elem,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Directional spectra

        Directional spectra have 0 frequencies. They can be of type point, line and area.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Line directional spectra
        """
    )
    return


@app.cell
def _(mikeio):
    fn_3 = '../tests/testdata/spectra/line_dir_spectra.dfsu'
    da_3 = mikeio.read(fn_3)[0]
    da_3
    return da_3, fn_3


@app.cell
def _(da_3):
    (da_3.n_frequencies, da_3.n_directions)
    return


@app.cell
def _(da_3):
    da5 = da_3.isel(time=0).isel(node=5)
    da5
    return (da5,)


@app.cell
def _(da5):
    da5.plot();
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

