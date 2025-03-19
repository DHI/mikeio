import marimo

__generated_with = "0.10.2"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        r"""
        # Dfsu - Export to shapefile

        1. Read selected item and timestep from dfsu
        2. Extract geometry
        3. Create GeoPandas dataframe
        4. Save to ESRI shapefile
        """
    )
    return


@app.cell
def _():
    import pandas as pd
    import geopandas as gpd
    import mikeio
    return gpd, mikeio, pd


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Step 1. read the selected data
        """
    )
    return


@app.cell
def _(mikeio):
    ds = mikeio.read("../tests/testdata/wind_north_sea.dfsu")
    ws = ds["Wind speed"][0]
    ws.plot();
    return ds, ws


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Step 2. extract geometry
        """
    )
    return


@app.cell
def _(ds):
    shp = ds.geometry.to_shapely()
    type(shp)
    return (shp,)


@app.cell
def _(mo):
    mo.md(
        r"""
        Geopandas does not like multipolygon, it should be a list of polygons
        """
    )
    return


@app.cell
def _(shp):
    poly_list = [p for p in shp.geoms]
    return (poly_list,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Step 3. Create a geopandas dataframe
        """
    )
    return


@app.cell
def _(pd, ws):
    df = pd.DataFrame({'wind_speed':ws.to_numpy()})
    df.head()
    return (df,)


@app.cell
def _(df, gpd, poly_list):
    gdf = gpd.GeoDataFrame(df,geometry=poly_list, crs=4326)
    return (gdf,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Step 4. Save to shapefile
        """
    )
    return


@app.cell
def _(gdf):
    gdf.to_file("wind_speed.shp")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Step 5...
        Do further work in QGIS

        ![QGIS](../images/dfsu_qgis.png)
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Would you prefer to have this workflow to be a method on the `mikeio.Dfsu` class?

        Post an issue on [GitHub](https://github.com/DHI/mikeio/issues) !
        """
    )
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

    files = ["wind_speed"]

    exts = ["cpg","dbf","shp","shx"]

    for file in files:
        for ext in exts:
            filename = f"{file}.{ext}"
            if os.path.exists(filename):
                os.remove(filename)
    return ext, exts, file, filename, files, os


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

