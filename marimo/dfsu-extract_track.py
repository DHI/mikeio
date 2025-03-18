import marimo

__generated_with = "0.11.21"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        r"""
        # Dfsu - Extract Track
        Similar to the MIKE tool DataTrackExtractionFM.exe the Dfsu method extract_track() can be used to extract model data along a track (e.g. satellite altimetry track)
        """
    )
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib_inline.backend_inline import set_matplotlib_formats
    set_matplotlib_formats('png')

    import mikeio
    return mikeio, np, pd, plt, set_matplotlib_formats


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Load dfsu result file
        The file contains surface elevation and wind speed model data. We wish to compare the model data with altimetry data
        """
    )
    return


@app.cell
def _():
    track_file = '../tests/testdata/altimetry_NorthSea_20171027.csv'
    data_file = '../tests/testdata/NorthSea_HD_and_windspeed.dfsu'
    return data_file, track_file


@app.cell
def _(data_file, mikeio):
    dfs = mikeio.open(data_file)
    dfs
    return (dfs,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Load and visualize altimetry tracks 
        The altimetry data is stored in a csv file. We plot the data on top of the dfsu mesh.
        """
    )
    return


@app.cell
def _(pd, track_file):
    track = pd.read_csv(track_file, index_col=0, parse_dates=True)
    return (track,)


@app.cell
def _(track):
    track.head()
    return


@app.cell
def _(dfs, track):
    ax = dfs.geometry.plot.mesh(figsize=(8,7), title="")
    track.plot.scatter('lon','lat', ax=ax)
    return (ax,)


@app.cell
def _(dfs, track):
    track_xy = track[['lon','lat']].values
    f'Inside domain: {sum(dfs.geometry.contains(track_xy))} points of the track (total: {len(track_xy)})'
    return (track_xy,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Extract track data from dfsu file
        The extract_track() takes a track definition (time, longitude, latitude of each point) as either a dataframe, a csv-file, a dfs0 file or a mikeio.Dataset.
        """
    )
    return


@app.cell
def _(dfs, track_file):
    e_track = dfs.extract_track(track_file)
    return (e_track,)


@app.cell
def _(e_track):
    e_track
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
