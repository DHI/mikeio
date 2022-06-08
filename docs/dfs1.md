# Dfs1

A dfs1 file contains node-based line series data. Dfs1 files do not contain enough metadata to determine their geographical position, but have a relative distance from the origo. 


```python
>>> import mikeio
>>> ds = mikeio.read("tide1.dfs1")
>>> ds
<mikeio.Dataset>
dims: (time:97, x:10)
time: 2019-01-01 00:00:00 - 2019-01-03 00:00:00 (97 records)
geometry: Grid1D (n=10, dx=0.06667)
items:
  0:  Level <Water Level> (meter)
```

## Grid 1D

The spatial information is available in the `geometry` attribute (accessible from Dfs1, Dataset, and DataArray), which in the case of a dfs1 file is a [`Grid1D`](Grid1D) geometry. 

```python
>>> ds.geometry
<mikeio.Grid1D>
x: [0, 0.06667, ..., 0.6] (nx=10, dx=0.06667)
```

Grid1D's primary properties and methods are: 

* `x` 
* `nx`
* `dx`
* `find_index()`
* `isel()`

See [API specification](Grid1D) below for details.




## Dfs1 API

```{eval-rst}
.. autoclass:: mikeio.Dfs1
	:members:
	:inherited-members:
```

## Grid1D API

```{eval-rst}
.. autoclass:: mikeio.Grid1D
	:members:
	:inherited-members:
```


## DataArray Plotter Grid1D API

A DataArray `da` with a Grid1D geometry can be plotted using `da.plot`. 

```{eval-rst}
.. autoclass:: mikeio.dataarray._DataArrayPlotterGrid1D
	:members:
	:inherited-members:
```