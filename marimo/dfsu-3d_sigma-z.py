import marimo

__generated_with = "0.10.2"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        r"""
        # Dfsu - 3D sigma-z
        This notebook demonstrates, reading from a sigma-z dfsu file, top- and bottom layers, extracting a profile, save selected layers to new dfsu file and save to mesh. 

        It also shows how to read a sigma-z vertical slice file. 
        """
    )
    return


@app.cell
def _():
    import mikeio
    return (mikeio,)


@app.cell
def _(filename, mikeio):
    _filename = '../tests/testdata/oresund_sigma_z.dfsu'
    dfs = mikeio.open(filename)
    dfs
    return (dfs,)


@app.cell
def _(dfs):
    dfs.geometry.plot.mesh();
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Save geometry to new mesh file
        Note that for sigma-z files, the mesh will be not match the original mesh in areas where z-layers are present! 
        """
    )
    return


@app.cell
def _(dfs):
    outmesh = "mesh_oresund_extracted.mesh"
    dfs.geometry.to_mesh(outmesh)
    return (outmesh,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Evaluate top layer
        """
    )
    return


@app.cell
def _(dfs):
    ds = dfs.read(layers="top")
    print(ds)
    max_t = ds['Temperature'].to_numpy().max()
    print(f'Maximum temperature in top layer: {max_t:.1f}')
    return ds, max_t


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Find position of max temperature and plot

        Find position of max temperature in first time step 
        """
    )
    return


@app.cell
def _(ds):
    timestep = 0
    elem = ds['Temperature'][timestep].to_numpy().argmax()
    max_x, max_y = ds.geometry.element_coordinates[elem,:2]
    print(f'Position of maximum temperature: (x,y) = ({max_x:.1f}, {max_y:.1f})')
    return elem, max_x, max_y, timestep


@app.cell
def _(ds, max_x, max_y, timestep):
    _ax = ds['Temperature'].isel(time=timestep).plot()
    _ax.plot(max_x, max_y, marker='*', markersize=20)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Read 1D profile from 3D file
        Find water column which has highest temperature and plot profile for all 3 time steps.
        """
    )
    return


@app.cell
def _(dfs, max_x, max_y):
    dsp = dfs.read(x=max_x, y=max_y) # select vertical column from dfsu-3d 
    dsp["Temperature"].plot();
    return (dsp,)


@app.cell
def _(mo):
    mo.md(
        r"""
        Note that the vertical column data is extrapolated to the bottom and surface! 

        The extrapolation can avoided using "extrapolate=False":
        """
    )
    return


@app.cell
def _(dsp):
    dsp["Temperature"].plot(extrapolate=False, marker='o');
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        If the data has more than a few timesteps, it can be more convenient to plot as 2d pcolormesh. We will simulate this by interpolating to 30min data. 

        Note that pcolormesh will plot using the static z information!
        """
    )
    return


@app.cell
def _(dsp):
    dspi = dsp.Salinity.interp_time(dt=1800)
    dspi.plot.pcolormesh();
    return (dspi,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Read top layer of a smaller area
        """
    )
    return


@app.cell
def _(dfs):
    bbox = [310000, 6192000, 380000, 6198000]
    ds_sub = dfs.read(area=bbox, layers="top")
    ds_sub
    return bbox, ds_sub


@app.cell
def _(ds_sub):
    ds_sub.Temperature.plot();
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Plot subset inside original model domain
        """
    )
    return


@app.cell
def _(dfs, ds_sub):
    _ax = ds_sub.Temperature.plot(figsize=(6, 7))
    dfs.geometry.plot.outline(ax=_ax, title=None)
    return


@app.cell
def _(ds_sub):
    ds_sub.to_dfs("oresund_data_extracted.dfsu")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # Select top 2 layers and write to new file
        get_layer_elements() can take a list of layers. Layers are counted positive from the bottom starting at 0 or alternatively counted negative from the top starting at -1. Here we take layers -1 and -2, i.e., the two top layers. 

        Next data is read from source file and finally written to a new dfsu file (which is now sigma-only dfsu file).
        """
    )
    return


@app.cell
def _(dfs):
    ds_top2 = dfs.read(layers=[-2, -1])
    ds_top2
    return (ds_top2,)


@app.cell
def _(ds_top2):
    outfile = "oresund_top2_layers.dfsu"
    ds_top2.to_dfs(outfile)
    return (outfile,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Read vertical slice (transect)
        """
    )
    return


@app.cell
def _(filename, mikeio):
    _filename = '../tests/testdata/oresund_vertical_slice.dfsu'
    ds_1 = mikeio.read(filename)
    ds_1
    return (ds_1,)


@app.cell
def _(ds_1):
    print(ds_1.geometry.bottom_elements[:9])
    print(ds_1.geometry.n_layers_per_column[:9])
    print(ds_1.geometry.top_elements[:9])
    return


@app.cell
def _(ds_1):
    ds_1.Temperature.plot()
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
    os.remove("mesh_oresund_extracted.mesh")
    os.remove("oresund_data_extracted.dfsu")
    os.remove("oresund_top2_layers.dfsu")
    return (os,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

