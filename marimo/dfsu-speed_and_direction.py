import marimo

__generated_with = "0.10.2"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        r"""
        # Dfsu - Speed and direction
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import mikeio
    return mikeio, np


@app.cell
def _(mikeio):
    ds = mikeio.read("../tests/testdata/HD2D.dfsu")
    ds
    return (ds,)


@app.cell
def _(mo):
    mo.md(
        r"""
        This file is missing current direction :-(

        Lets'fix that!
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Calculate speed & direction
        """
    )
    return


@app.cell
def _(ds):
    ds.U_velocity
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        In order to use Numpy functions on a DataArray, we first convert the DataArrays (U, V) to standard NumPy ndarrays.
        """
    )
    return


@app.cell
def _(ds):
    u = ds.U_velocity.to_numpy()
    v = ds.V_velocity.to_numpy()
    return u, v


@app.cell
def _(np, u, v):
    direction = np.mod(90 -np.rad2deg(np.arctan2(v,u)),360)
    return (direction,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Write new file
        """
    )
    return


@app.cell
def _(direction, ds, mikeio):
    from mikeio.eum import ItemInfo, EUMUnit, EUMType

    ds["Current direction"] = mikeio.DataArray(direction, time= ds.time, item = ItemInfo("Current direction", EUMType.Current_Direction, EUMUnit.degree), geometry=ds.geometry)
    ds
    return EUMType, EUMUnit, ItemInfo


@app.cell
def _(ds):
    ds.to_dfs("speed_direction.dfsu")
    return


@app.cell
def _(mikeio):
    nds = mikeio.read("speed_direction.dfsu")
    nds
    return (nds,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Plot
        """
    )
    return


@app.cell
def _(ds):
    step = 1
    _ax = ds.Current_speed[step].plot(figsize=(10, 10))
    _ax.set_ylim([None, 6903000])
    _ax.set_xlim([607000, None])
    ec = ds.geometry.element_coordinates
    x = ec[:, 0]
    y = ec[:, 1]
    u_1 = ds.U_velocity.to_numpy()
    v_1 = ds.V_velocity.to_numpy()
    _ax.quiver(x, y, u_1[step], v_1[step], scale=6, minshaft=3)
    return ec, step, u_1, v_1, x, y


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Plot quiver on Cartesian overlay instead 
        Create overset grid and interpolate data on to this
        """
    )
    return


@app.cell
def _(ds):
    g = ds.geometry.get_overset_grid(dx=50)
    g
    return (g,)


@app.cell
def _(g):
    g.projection
    return


@app.cell
def _(ds, g):
    ui = ds.U_velocity.interp_like(g)
    vi = ds.V_velocity.interp_like(g)
    return ui, vi


@app.cell
def _(ui):
    ui.plot();
    return


@app.cell
def _(ds, g, step, ui, vi):
    _ax = ds.Current_speed.plot(figsize=(10, 10))
    u_2 = ui.to_numpy()
    v_2 = vi.to_numpy()
    _ax.quiver(g.x, g.y, u_2[step], v_2[step], scale=8, minshaft=5)
    _ax.set_ylim([None, 6903000])
    _ax.set_xlim([607000, None])
    _ax.set_title(f'Current speed with overset grid; {ds.time[step]}')
    _ax.set_xlabel('Easting (m)')
    _ax.set_ylabel('Northing (m)')
    return u_2, v_2


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Clean up
        """
    )
    return


@app.cell
def _():
    import os
    os.remove("speed_direction.dfsu")
    return (os,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

