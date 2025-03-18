import marimo

__generated_with = "0.10.2"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        r"""
        # Boundary 

        The task is to combine current velocities from an oceanographic model without tides with tidal current.

        * The oceanographic model data is vertically resolved and available in a vertical transect as a Dfs2 with daily timestep
        * The tidal model is vertically integrated and available in a line transect as a Dfs1 with hourly timestep
        * The spatial grid is identical

        ## Steps
        1. Read data
        2. Interpolate in time
        3. Calculate $\mathbf{U}_{combined} = \mathbf{U}_{tide} + \mathbf{U}_{residual}$
        4. Write to new dfs2
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
    ds_surge = mikeio.read("../tests/testdata/uv_vertical_daily.dfs2")
    ds_surge
    return (ds_surge,)


@app.cell
def _(mikeio):
    ds_tide = mikeio.read("../tests/testdata/vu_tide_hourly.dfs1")
    ds_tide
    return (ds_tide,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ### 1. Interpolate in time
        """
    )
    return


@app.cell
def _(ds_surge, ds_tide):
    ds_surge_h = ds_surge.interp_time(ds_tide)
    ds_surge_h
    return (ds_surge_h,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ### 2. Combine

        **Note that the naming and order is inconsistent between the two data sources!**
        """
    )
    return


@app.cell
def _(ds_tide, np):
    u_tide = ds_tide["Tidal current component (geographic East)"].to_numpy()
    u_tide = np.expand_dims(u_tide, 1)
    u_tide.shape
    return (u_tide,)


@app.cell
def _(ds_surge_h, mikeio, u_tide):
    u_surge = ds_surge_h["eastward_sea_water_velocity"]
    u_combined = u_surge + u_tide
    u_combined.item = mikeio.ItemInfo("U", mikeio.EUMType.u_velocity_component)
    return u_combined, u_surge


@app.cell
def _(ds_surge_h, ds_tide, mikeio, np):
    v_tide = ds_tide["Tidal current component (geographic North)"].to_numpy()
    v_tide = np.expand_dims(v_tide, 1)
    v_surge = ds_surge_h["northward_sea_water_velocity"]
    v_combined = v_surge + v_tide
    v_combined.item = mikeio.ItemInfo("V", mikeio.EUMType.u_velocity_component)
    return v_combined, v_surge, v_tide


@app.cell
def _(mo):
    mo.md(
        r"""
        ### 3. Write to dfs2
        """
    )
    return


@app.cell
def _(mikeio, u_combined, v_combined):
    ds_combined = mikeio.Dataset([u_combined, v_combined])
    return (ds_combined,)


@app.cell
def _(ds_combined):
    ds_combined_1 = ds_combined.dropna()
    ds_combined_1
    return (ds_combined_1,)


@app.cell
def _(ds_combined_1):
    ds_combined_1.to_dfs('uv_combined.dfs2')
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Clean up

        """
    )
    return


@app.cell
def _():
    import os
    os.remove("uv_combined.dfs2")
    return (os,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

