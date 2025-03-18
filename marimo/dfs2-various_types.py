import marimo

__generated_with = "0.10.2"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        r"""
        # Dfs2 - Various types

        This notebook examines various Dfs2 types: 

        * Horizontal
            - UTM (utm_not_rotated_neurope_temp.dfs2)
            - Long/Lat (europe_wind_long_lat.dfs2)
            - Long/Lat global (global_long_lat_pacific_view_temperature_delta.dfs2)
            - Local coordinates (M3WFM_sponge_local_coordinates.dfs2)
        * Rotated 
            - UTM (BW_Ronne_Layout1998_rotated.dfs2)    
        * Vertical (hd_vertical_slice.dfs2)
        * Spectral
            - Linear f-axis (dir_wave_analysis_spectra.dfs2)
            - Logarithmic f-axis (pt_spectra.dfs2)

        For each of these types, it's possible to :

        * plot
        * isel 
        * sel (point, line or area)
        * read and write without changing header (origo and rotation)
        """
    )
    return


@app.cell
def _():
    import mikeio
    return (mikeio,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Horizontal
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Horizontal, UTM (projected)
        """
    )
    return


@app.cell
def _(dfs, fn, mikeio):
    _fn = '../tests/testdata/utm_not_rotated_neurope_temp.dfs2'
    _dfs = mikeio.open(fn)
    da = dfs.read()[0]
    da
    return (da,)


@app.cell
def _(da):
    da.geometry
    return


@app.cell
def _(da):
    da.plot();
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Horizontal, geographic (long/lat)
        """
    )
    return


@app.cell
def _(fn, mikeio):
    _fn = '../tests/testdata/europe_wind_long_lat.dfs2'
    da_1 = mikeio.read(fn)[1]
    da_1
    return (da_1,)


@app.cell
def _(da_1):
    da_1.geometry
    return


@app.cell
def _(da_1):
    da_1.plot()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Horizontal, geographic with global coverage
        """
    )
    return


@app.cell
def _(fn, mikeio):
    _fn = '../tests/testdata/global_long_lat_pacific_view_temperature_delta.dfs2'
    da_2 = mikeio.read(fn)[0]
    da_2
    return (da_2,)


@app.cell
def _(da_2):
    da_2.geometry
    return


@app.cell
def _(da_2):
    da_2.plot()
    return


@app.cell
def _(da_2):
    da_2.sel(area=[110, -50, 160, -10]).plot()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Local coordinates
        """
    )
    return


@app.cell
def _(mikeio):
    import numpy as np
    da_3 = mikeio.DataArray(np.array([[1, 2, 3], [4, 5, 6]]), geometry=mikeio.Grid2D(nx=3, ny=2, dx=0.5, projection='NON-UTM'))
    da_3.plot()
    return da_3, np


@app.cell
def _(fn, mikeio):
    _fn = '../tests/testdata/M3WFM_sponge_local_coordinates.dfs2'
    da_4 = mikeio.read(fn)[0]
    da_4
    return (da_4,)


@app.cell
def _(da_4):
    da_4.plot()
    return


@app.cell
def _(da_4):
    da_4.sel(y=3).plot()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Rotated
        """
    )
    return


@app.cell
def _(fn, mikeio):
    _fn = '../tests/testdata/BW_Ronne_Layout1998_rotated.dfs2'
    da_5 = mikeio.read(fn)[0]
    da_5
    return (da_5,)


@app.cell
def _(da_5):
    da_5.geometry
    return


@app.cell
def _(da_5):
    da_5.plot()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Vertical
        """
    )
    return


@app.cell
def _(fn, mikeio):
    _fn = '../tests/testdata/hd_vertical_slice.dfs2'
    da_6 = mikeio.open(fn, type='vertical').read()[0]
    da_6
    return (da_6,)


@app.cell
def _(da_6):
    da_6.geometry
    return


@app.cell
def _(da_6):
    da_6.plot()
    return


@app.cell
def _(da_6):
    da_6.isel(y=slice(45, None)).plot()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Spectral

        When reading spectral dfs2 files, the user must specify type='spectral'. 
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Linear f-axis 
        """
    )
    return


@app.cell
def _(dfs, fn, mikeio):
    _fn = '../tests/testdata/spectra/dir_wave_analysis_spectra.dfs2'
    _dfs = mikeio.open(fn, type='spectral')
    da_7 = dfs.read()[0]
    da_7
    return (da_7,)


@app.cell
def _(da_7):
    da_7.geometry
    return


@app.cell
def _(da_7):
    da_7.plot()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Logarithmic f-axis
        """
    )
    return


@app.cell
def _(dfs, fn, mikeio):
    _fn = '../tests/testdata/spectra/pt_spectra.dfs2'
    _dfs = mikeio.open(fn, type='spectral')
    da_8 = dfs.read()[0]
    da_8
    return (da_8,)


@app.cell
def _(da_8):
    da_8.geometry.is_spectral
    return


@app.cell
def _(da_8):
    da_8.geometry.x
    return


@app.cell
def _(da_8):
    da_8.geometry
    return


@app.cell
def _(da_8):
    da_8.plot()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

