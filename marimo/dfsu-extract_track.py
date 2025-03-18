import marimo

__generated_with = "0.10.2"
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
        # Load dfsu result file
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
        # Load and visualize altimetry tracks 
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
    ax = dfs.geometry.plot.mesh(figsize=(8,7))
    track.plot.scatter('lon','lat', ax=ax);
    return (ax,)


@app.cell
def _(dfs, track):
    track_xy = track[['lon','lat']].values
    print(f'Inside domain: {sum(dfs.geometry.contains(track_xy))} points of the track (total: {len(track_xy)})')
    return (track_xy,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Extract track data from dfsu file
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
    # convert to dataframe and rename columns
    df = e_track.to_dataframe()
    df.columns = ['Longitude', 'Latitude', 'Model_surface_elevation', 'Model_wind_speed']
    return (df,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Compare with the observed altimetry values
        """
    )
    return


@app.cell
def _(df, track):
    df['Obs_surface_elevation'] = track['surface_elevation']
    df['Obs_wind_speed'] = track['wind_speed']
    df.dropna(inplace=True)
    return


@app.cell
def _(df, np, resi):
    _resi = df.Model_wind_speed - df.Obs_wind_speed
    _bias = resi.median()
    _rmse = np.sqrt(np.mean(resi ** 2))
    print(f'Wind speed: bias={_bias:.2f}m/s, rmse={_rmse:.2f}m/s')
    return


@app.cell
def _(df, plt):
    df.plot.scatter('Obs_wind_speed','Model_wind_speed')
    plt.plot([0,25],[0,25], color='r')
    plt.gca().set_aspect('equal')
    return


@app.cell
def _(df, np, resi):
    _resi = df.Model_surface_elevation - df.Obs_surface_elevation
    _bias = resi.median()
    _rmse = np.sqrt(np.mean(resi ** 2))
    print(f'Surface elevation: bias={100 * _bias:.2f}cm, rmse={100 * _rmse:.2f}cm')
    return


@app.cell
def _(df, plt):
    df.plot.scatter('Obs_surface_elevation','Model_surface_elevation')
    plt.plot([-0.6,2.5],[-0.6,2.5], color='r')
    plt.gca().set_aspect('equal')
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

