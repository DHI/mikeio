import marimo

__generated_with = "0.10.2"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        r"""
        # Generic dfs processing
        Tools and methods that applies to any type of dfs files. 

        * mikeio.read()
        * mikeio.generic: methods that read any dfs file and outputs a new dfs file of the same type
           - concat: Concatenates files along the time axis  
           - scale: Apply scaling to any dfs file
           - sum: Sum two dfs files 
           - diff: Calculate difference between two dfs files
           - extract: Extract timesteps and/or items to a new dfs file
           - time-avg: Create a temporally averaged dfs file
           - quantile: Create temporal quantiles of dfs file

        """
    )
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import mikeio
    import mikeio.generic
    return mikeio, plt


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Concatenation
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Take a look at these two files with overlapping timesteps.
        """
    )
    return


@app.cell
def _(mikeio):
    t1 = mikeio.read("../tests/testdata/tide1.dfs1")
    t1
    return (t1,)


@app.cell
def _(mikeio):
    t2 = mikeio.read("../tests/testdata/tide2.dfs1")
    t2
    return (t2,)


@app.cell
def _(mo):
    mo.md(
        r"""
        Plot one of the points along the line.
        """
    )
    return


@app.cell
def _(plt, t1, t2):
    plt.plot(t1.time,t1[0].isel(x=1).values, label="File 1")
    plt.plot(t2.time,t2[0].isel(x=1).values,'k+', label="File 2")
    plt.legend()
    return


@app.cell
def _(mikeio):
    mikeio.generic.concat(infilenames=["../tests/testdata/tide1.dfs1",
                                       "../tests/testdata/tide2.dfs1"],
                         outfilename="concat.dfs1")
    return


@app.cell
def _(mikeio):
    c = mikeio.read("concat.dfs1")
    c[0].isel(x=1).plot()
    c
    return (c,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Difference between two files

        Take difference between two dfs files with same structure - e.g. to see the difference in result between two calibration runs
        """
    )
    return


@app.cell
def _(mikeio):
    fn1 = "../tests/testdata/oresundHD_run1.dfsu"
    fn2 = "../tests/testdata/oresundHD_run2.dfsu"
    fn_diff = "oresundHD_difference.dfsu"
    mikeio.generic.diff(fn1, fn2, fn_diff)
    return fn1, fn2, fn_diff


@app.cell
def _(fn1, fn2, fn_diff, mikeio, plt):
    _, ax = plt.subplots(1,3, sharey=True, figsize=(12,5))
    da = mikeio.read(fn1, time=-1)[0]
    da.plot(vmin=0.06, vmax=0.27, ax=ax[0], title='run 1')
    da = mikeio.read(fn2, time=-1)[0]
    da.plot(vmin=0.06, vmax=0.27, ax=ax[1], title='run 2')
    da = mikeio.read(fn_diff, time=-1)[0]
    da.plot(vmin=-0.1, vmax=0.1, cmap='coolwarm', ax=ax[2], title='difference');
    return ax, da


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Extract time steps or items

        The extract() method can extract a part of a file:

        * **time slice** by specifying *start* and/or *end*
        * specific **items**
        """
    )
    return


@app.cell
def _(mikeio):
    _infile = '../tests/testdata/tide1.dfs1'
    mikeio.generic.extract(_infile, 'extracted.dfs1', start='2019-01-02')
    return


@app.cell
def _(mikeio):
    _e = mikeio.read('extracted.dfs1')
    _e
    return


@app.cell
def _(mikeio):
    _infile = '../tests/testdata/oresund_vertical_slice.dfsu'
    mikeio.generic.extract(_infile, 'extracted.dfsu', items='Salinity', end=-2)
    return


@app.cell
def _(mikeio):
    _e = mikeio.read('extracted.dfsu')
    _e
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Scaling

        Adding a constant e.g to adjust datum
        """
    )
    return


@app.cell
def _(mikeio):
    ds = mikeio.read("../tests/testdata/gebco_sound.dfs2")
    ds.Elevation[0].plot();
    return (ds,)


@app.cell
def _(ds):
    ds['Elevation'][0,104,131].to_numpy()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        This is the processing step.
        """
    )
    return


@app.cell
def _(mikeio):
    mikeio.generic.scale("../tests/testdata/gebco_sound.dfs2", 
                         "gebco_sound_local_datum.dfs2",
                         offset=-2.1
                         )
    return


@app.cell
def _(mikeio):
    ds2 = mikeio.read("gebco_sound_local_datum.dfs2")
    ds2['Elevation'][0].plot()
    return (ds2,)


@app.cell
def _(ds2):
    ds2['Elevation'][0,104,131].to_numpy()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Spatially varying correction
        """
    )
    return


@app.cell
def _(ds):
    import numpy as np
    factor = np.ones_like(ds['Elevation'][0].to_numpy())
    factor.shape
    return factor, np


@app.cell
def _(mo):
    mo.md(
        r"""
        Add some spatially varying factors, exaggerated values for educational purpose.
        """
    )
    return


@app.cell
def _(factor, plt):
    factor[:,0:100] = 5.3
    factor[0:40,] = 0.1
    factor[150:,150:] = 10.7
    plt.imshow(factor)
    plt.colorbar();
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        The 2d array must first be flipped upside down and then converted to a 1d vector using [numpy.ndarray.flatten](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html) to match how data is stored in dfs files.
        """
    )
    return


@app.cell
def _(factor, mikeio, np):
    factor_ud = np.flipud(factor)
    factor_vec  = factor_ud.flatten()
    mikeio.generic.scale("../tests/testdata/gebco_sound.dfs2", 
                         "gebco_sound_spatial.dfs2",
                         factor=factor_vec
                         )
    return factor_ud, factor_vec


@app.cell
def _(mikeio):
    ds3 = mikeio.read("gebco_sound_spatial.dfs2")
    ds3.Elevation[0].plot();
    return (ds3,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Time average
        """
    )
    return


@app.cell
def _(mikeio):
    fn = "../tests/testdata/NorthSea_HD_and_windspeed.dfsu"
    fn_avg = "Avg_NorthSea_HD_and_windspeed.dfsu"
    mikeio.generic.avg_time(fn, fn_avg)
    return fn, fn_avg


@app.cell
def _(fn, mikeio):
    ds_1 = mikeio.read(fn)
    ds_1.mean(axis=0).describe()
    return (ds_1,)


@app.cell
def _(fn_avg, mikeio):
    ds_avg = mikeio.read(fn_avg)
    ds_avg.describe()
    return (ds_avg,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Quantile

        Example that calculates the 25%, 50% and 75% percentile for all items in a dfsu file.
        """
    )
    return


@app.cell
def _(mikeio):
    fn_1 = '../tests/testdata/NorthSea_HD_and_windspeed.dfsu'
    fn_q = 'Q_NorthSea_HD_and_windspeed.dfsu'
    mikeio.generic.quantile(fn_1, fn_q, q=[0.25, 0.5, 0.75])
    return fn_1, fn_q


@app.cell
def _(fn_q, mikeio):
    ds_2 = mikeio.read(fn_q)
    ds_2
    return (ds_2,)


@app.cell
def _(ds_2):
    da_q75 = ds_2['Quantile 0.75, Wind speed']
    da_q75.plot(title='75th percentile, wind speed', label='m/s')
    return (da_q75,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Clean up
        """
    )
    return


@app.cell
def _(fn_q):
    import os
    os.remove("concat.dfs1")
    os.remove("oresundHD_difference.dfsu")
    os.remove("extracted.dfs1")
    os.remove("extracted.dfsu")
    os.remove("gebco_sound_local_datum.dfs2")
    os.remove("gebco_sound_spatial.dfs2")
    os.remove("Avg_NorthSea_HD_and_windspeed.dfsu")
    os.remove(fn_q)
    return (os,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

