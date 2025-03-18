import marimo

__generated_with = "0.10.2"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        r"""
        # Dfsu - Vertical Profile
        This notebooks demonstrates plotting of vertical profile (transect) dfsu. 
        """
    )
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import mikeio
    return mikeio, plt


@app.cell
def _(filename, mikeio):
    _filename = '../tests/testdata/oresund_vertical_slice.dfsu'
    ds = mikeio.read(filename)
    ds
    return (ds,)


@app.cell
def _(ds):
    g = ds.geometry
    g
    return (g,)


@app.cell
def _(g):
    import numpy as np
    ec2d = g.element_coordinates[g.top_elements,:2]
    xe, ye = ec2d[:,0], ec2d[:,1]
    np.argmin((xe - 359615.47172605) ** 2 + (ye - 6.145e+06) ** 2)
    return ec2d, np, xe, ye


@app.cell
def _(g):
    g._find_nearest_element_2d([359615,6.145e+06])
    return


@app.cell
def _(ds):
    ds.sel(x=359615, y=6.145e+06, z=-3).plot()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        The geometry can be visualized from above (to be shown on a map) using g.plot() and from the side showing the 2dv transect mesh with g.plot.mesh(). 

        Let's show the transect on top of the model domain...
        """
    )
    return


@app.cell
def _(mikeio):
    dfs = mikeio.open("../tests/testdata/oresundHD_run1.dfsu")
    model_domain = dfs.geometry
    return dfs, model_domain


@app.cell
def _(g, model_domain, plt):
    _, ax = plt.subplots(1, 2, figsize=(12, 4))
    model_domain.plot(ax=_ax[0], title='Transect')
    g.plot(color='r', ax=_ax[0])
    g.plot.mesh(ax=_ax[1], title='Transect mesh')
    return (ax,)


@app.cell
def _(mo):
    mo.md(
        r"""
        We would like to show two points of interest A and B on the map. The geometry object has a method for finding the nearest relative position...
        """
    )
    return


@app.cell
def _(g):
    ptA = [3.55e+05,  6.145e+06]
    ptB = [3.62e+05,  6.166e+06] 
    distA = g.get_nearest_relative_distance(ptA)
    distB = g.get_nearest_relative_distance(ptB)
    distA, distB
    return distA, distB, ptA, ptB


@app.cell
def _(mo):
    mo.md(
        r"""
        Let's now visualize the points on the map and transect
        """
    )
    return


@app.cell
def _(distA, distB, g, model_domain, plt, ptA, ptB):
    _, ax_1 = plt.subplots(1, 2, figsize=(12, 4))
    model_domain.plot(ax=_ax[0], title='Transect')
    g.plot(color='r', ax=_ax[0])
    _ax[0].plot(*ptA, color='b', marker='*', markersize=10)
    _ax[0].plot(*ptB, color='b', marker='*', markersize=10)
    g.plot.mesh(ax=_ax[1], title='Transect mesh')
    _ax[1].axvline(distA, color='0.5')
    _ax[1].text(distA + 500, -20, 'position A')
    _ax[1].axvline(distB, color='0.5')
    _ax[1].text(distB + 500, -20, 'position B')
    return (ax_1,)


@app.cell
def _(distA, distB, ds):
    _ax = ds.Temperature.isel(time=2).plot(figsize=(12, 4))
    _ax.axvline(distA, color='0.5')
    _ax.text(distA + 500, -20, 'position A')
    _ax.axvline(distB, color='0.5')
    _ax.text(distB + 500, -20, 'position B')
    return


@app.cell
def _(ds, plt):
    time_step = 1
    fig, ax_2 = plt.subplots(2, 1, figsize=(10, 8))
    ds.Temperature[time_step].plot(ax=_ax[0])
    ds.Salinity[time_step].plot(ax=_ax[1], title=None)
    return ax_2, fig, time_step


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Kalundborg case
        A non-straight vertical profile (transect) from a model in geographical coordinates.
        """
    )
    return


@app.cell
def _(filename, mikeio):
    _filename = '../tests/testdata/kalundborg_coarse.mesh'
    model_domain_1 = mikeio.open(filename).geometry
    _filename = '../tests/testdata/kalundborg_transect.dfsu'
    ds_1 = mikeio.read(filename)
    ds_1
    return ds_1, model_domain_1


@app.cell
def _(ds_1, model_domain_1):
    _ax = model_domain_1.plot.outline()
    ds_1.geometry.plot(color='cyan', ax=_ax)
    return


@app.cell
def _(ds_1):
    ds_1.U_velocity.plot(figsize=(12, 4))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Spatial subsetting 

        Both points and parts of the 2dv domain can selected.
        """
    )
    return


@app.cell
def _(ds_1):
    ptA_1 = [10.8, 55.6, -3]
    ds_1.geometry.get_nearest_relative_distance(ptA_1)
    return (ptA_1,)


@app.cell
def _(mo):
    mo.md(
        r"""
        Points can be extracted:
        """
    )
    return


@app.cell
def _(ds_1, ptA_1):
    ds_pt = ds_1.sel(x=ptA_1[0], y=ptA_1[1], z=ptA_1[2])
    ds_pt.plot()
    return (ds_pt,)


@app.cell
def _(mo):
    mo.md(
        r"""
        And vertical columns...
        """
    )
    return


@app.cell
def _(ds_1, plt, ptA_1):
    u_col = ds_1.sel(x=ptA_1[0], y=ptA_1[1]).U_velocity
    u_col.plot()
    plt.legend(ds_1.time)
    return (u_col,)


@app.cell
def _(mo):
    mo.md(
        r"""
        Or parts of the 2dv transect... here selecting the part with relative distance between 10 and 25 km
        """
    )
    return


@app.cell
def _(ds_1, idx, np):
    rd = ds_1.geometry.relative_element_distance
    _idx = np.where(np.logical_and(10000 < rd, rd < 25000))[0]
    dssub = ds_1.isel(element=idx)
    dssub
    return dssub, rd


@app.cell
def _(dssub):
    dssub.Temperature.plot();
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Or specific layers: 
        """
    )
    return


@app.cell
def _(ds_1, idx):
    _idx = ds_1.geometry.find_index(layers=range(-6, -1))
    dssub_1 = ds_1.isel(element=idx)
    dssub_1
    return (dssub_1,)


@app.cell
def _(dssub_1):
    dssub_1.Temperature.plot(figsize=(12, 3))
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

