
# Dfs2

A dfs2 file is also called a grid series file. Values in a dfs2 file are ‘element based’, i.e. values are defined in the centre of each grid cell. 


```python
>>> import mikeio
>>> ds = mikeio.read("gebco_sound.dfs2")
>>> ds
<mikeio.Dataset>
dims: (time:1, y:264, x:216)
time: 2020-05-15 11:04:52 (time-invariant)
geometry: Grid2D (ny=264, nx=216)
items:
  0:  Elevation <Total Water Depth> (meter)
```


## Grid2D

The spatial information is available in the `geometry` attribute (accessible from Dfs2, Dataset, and DataArray), which in the case of a dfs2 file is a [`Grid2D`](Grid2D) geometry. 

```python
>>> ds.geometry
<mikeio.Grid2D>
x: [12.2, 12.21, ..., 13.1] (nx=216, dx=0.004167)
y: [55.2, 55.21, ..., 56.3] (ny=264, dy=0.004167)
projection: LONG/LAT
```

Grid2D's primary properties and methods are: 

* `x` 
* `nx`
* `dx`
* `y`
* `ny`
* `dy`
* `origin`
* `projection`
* `xy` 
* `bbox`
* `contains()`
* `find_index()`
* `isel()`
* `to_mesh()`

See [API specification](Grid2D) below for details.


## Dfs2 Example notebooks

* [Dfs2 | getting-started-with-mikeio](https://dhi.github.io/getting-started-with-mikeio/dfs2.html)
* [Dfs2-Bathymetry](https://nbviewer.jupyter.org/github/DHI/mikeio/blob/main/notebooks/Dfs2%20-%20Bathymetry.ipynb) - GEBCO NetCDF/xarray to dfs2 
* [Dfs2-Boundary](https://nbviewer.jupyter.org/github/DHI/mikeio/blob/main/notebooks/Dfs2%20-%20Boundary.ipynb) - Vertical transect dfs2, interpolation in time 
* [Dfs2-Export-to-netCDF](https://nbviewer.jupyter.org/github/DHI/mikeio/blob/main/notebooks/Dfs2%20-%20Export%20to%20netcdf.ipynb) Export dfs2 to NetCDF
* [Dfs2-GFS](https://nbviewer.jupyter.org/github/DHI/mikeio/blob/main/notebooks/Dfs2%20-%20Global%20Forecasting%20System.ipynb) - GFS NetCDF/xarray to dfs2
* [Dfs2-SST](https://nbviewer.jupyter.org/github/DHI/mikeio/blob/main/notebooks/Dfs2%20-%20Sea%20surface%20temperature.ipynb) - DMI NetCDF/xarray to dfs2 



## Dfs2 API

```{eval-rst}
.. autoclass:: mikeio.Dfs2
	:members:
	:inherited-members:
```

## Grid2D API

```{eval-rst}
.. autoclass:: mikeio.Grid2D
	:members:
	:inherited-members:
```

## DataArray Plotter Grid2D API

A DataArray `da` with a Grid2D geometry can be plotted using `da.plot`. 

```{eval-rst}
.. autoclass:: mikeio.dataarray._DataArrayPlotterGrid2D
	:members:
	:inherited-members:
```