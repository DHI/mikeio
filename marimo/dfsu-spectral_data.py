import marimo

__generated_with = "0.11.21"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        r"""
        # Dfsu - spectral data

        MIKE 21 SW can output full spectral information in points, along lines or in an area. In all these cases data are stored in dfsu files with additional axes: frequency and directions. 

        This notebook explores reading __full__ spectral dfsu files from MIKE 21 SW as 

        * point
        * line
        * area
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
def _(mo):
    mo.md(r"""## Read dfsu point spectrum""")
    return


@app.cell
def _(mikeio):
    da = mikeio.read('../tests/testdata/spectra/pt_spectra.dfsu')[0]
    da
    return (da,)


@app.cell
def _(da):
    da.plot() # plots first timestep by default
    return


@app.cell
def _(mo):
    mo.md(r"""Don't like the default plot? No worries, it can be customized.""")
    return


@app.cell
def _(da, np):
    ax = da.plot.patch(rmax=8)
    dird = np.round(da.directions, 2)
    ax.set_thetagrids(dird, labels=dird)
    ax
    return ax, dird


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Dfsu line spectrum

        Data in dfsu line spectra is node-based contrary to must other dfsu-formats.
        """
    )
    return


@app.cell
def _(mikeio):
    da_1 = mikeio.read('../tests/testdata/spectra/line_spectra.dfsu').Energy_density
    da_1
    return (da_1,)


@app.cell
def _(da_1):
    spec = da_1[0].isel(node=3)
    spec
    return (spec,)


@app.cell
def _(spec):
    spec.plot(cmap="Greys", rmax=8, r_as_periods=True)
    return


@app.cell
def _(mo):
    mo.md(r"""### Plot Hm0 on a line""")
    return


@app.cell
def _(da_1):
    Hm0 = da_1.isel(time=0).to_Hm0()
    Hm0.plot(title='Hm0 on a line crossing the domain')
    return (Hm0,)


@app.cell
def _(mo):
    mo.md(r"""## Dfsu area spectrum""")
    return


@app.cell
def _(mikeio):
    da_2 = mikeio.read('../tests/testdata/spectra/area_spectra.dfsu', items='Energy density')[0]
    da_2
    return (da_2,)


@app.cell
def _(da_2):
    da_2.plot()
    return


@app.cell
def _(da_2):
    da_pt = da_2.sel(x=2.9, y=52.5)
    da_pt
    return (da_pt,)


@app.cell
def _(da_pt):
    da_pt.plot(rmax=9);
    return


@app.cell
def _(mo):
    mo.md(r"""### Interactive widget for exploring spectra in different points""")
    return


@app.cell
def _():
    from datetime import timedelta
    return (timedelta,)


@app.cell
def _(da_2, mo):
    el = mo.ui.slider(start=0, stop=(da_2.geometry.n_elements -1), label="Element")
    t = mo.ui.slider(start=0, stop=(da_2.n_timesteps -1), label="Time")
    [el,t]
    return el, t


@app.cell
def _(da_2, el, t, timedelta):
    id = el.value
    step = t.value

    dspec = da_2[step, id]
    time = da_2.start_time + timedelta(seconds=step * da_2.timestep)
    dspec.plot(vmax=0.04, vmin=0, rmax=8, title=f'Wave spectrum, {time}, element: {id}')
    return dspec, id, step, time


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
