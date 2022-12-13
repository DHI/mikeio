# DataArray

The [DataArray](DataArray) is the common MIKE IO data structure 
for *item* data from dfs files. 
The {py:meth}`mikeio.read` methods returns a Dataset as a container of DataArrays (Dfs items)

Each DataArray have the following properties:
* **item** - an  {py:class}`mikeio.eum.ItemInfo` with name, type and unit
* **time** - a {py:class}`pandas.DatetimeIndex` with the time instances of the data
* **geometry** - a Geometry object with the spatial description of the data
* **values** - a {py:class}`numpy.ndarray`

Use DataArray's string representation to get an overview of the DataArray


```python
>>> import mikeio
>>> ds = mikeio.read("testdata/HD2D.dfsu")
>>> da = ds["Surface Elevation"]
>>> da
<mikeio.DataArray>
name: Surface elevation
dims: (time:9, element:884)
time: 1985-08-06 07:00:00 - 1985-08-07 03:00:00 (9 records)
geometry: Dfsu2D (884 elements, 529 nodes)
```


## Temporal selection

```python
>>> da.sel(time="1985-08-06 12:00")
<mikeio.DataArray>
name: Surface elevation
dims: (element:884)
time: 1985-08-06 12:00:00 (time-invariant)
geometry: Dfsu2D (884 elements, 529 nodes)
values: [0.1012, 0.1012, ..., 0.105]

>>> da["1985-8-7":]
<mikeio.DataArray>
name: Surface elevation
dims: (time:2, element:884)
time: 1985-08-07 00:30:00 - 1985-08-07 03:00:00 (2 records)
geometry: Dfsu2D (884 elements, 529 nodes)

```

## Spatial selection

The `sel` method finds the nearest element.

```python
>>> da.sel(x=607002, y=6906734)
<mikeio.DataArray>
name: Surface elevation
dims: (time:9)
time: 1985-08-06 07:00:00 - 1985-08-07 03:00:00 (9 records)
geometry: GeometryPoint2D(x=607002.7094112666, y=6906734.833048992)
values: [0.4591, 0.8078, ..., -0.6311]
```

## Modifying values

You can modify the values of a DataArray by changing its `values`: 

```python
>>> da.values[0, 3] = 5.0
```

If you wish to change values of a subset of the DataArray you should be aware of the difference between a _view_ and a _copy_ of the data. Similar to NumPy, MIKE IO selection method will return a _view_ of the data when using single index and slices, but a _copy_ of the data using fancy indexing (a list of indicies or boolean indexing). Note that prior to release 1.3, MIKE IO would always return a copy. 

It is recommended to change the values using `values` property directly on the original DataArray (like above), but it is also possible to change the values of the original DataArray by working on a subset DataArray if it is selected with single index or slice as explained above. 

```python
>>> da_sub = da.isel(time=0)
>>> da_sub.values[:] = 5.0    # will change da
```

Fancy indexing will return a _copy_ and therefore not change the original:

```python
>>> da_sub = da.isel(time=[0,1,2])
>>> da_sub.values[:] = 5.0    # will NOT change da
```




## Plotting

The plotting of a DataArray is context-aware meaning that plotting behaviour depends on the geometry of the DataArray being plotted. 

```python
>>> da = mikeio.read("testdata/HD2D.dfsu")["Surface Elevation"]
>>> da.plot()
>>> da.plot.contourf()
>>> da.plot.mesh()
```

See details in the [API specification](_DatasetPlotter) below and in the bottom of the relevant pages e.g. [DataArray Plotter Grid2D API](_DataArrayPlotterGrid2D) on the dfs2 page.



## Properties

The DataArray has several properties:

* n_items - Number of items
* n_timesteps - Number of timesteps
* n_elements - Number of elements
* start_time - First time instance (as datetime)
* end_time - Last time instance (as datetime)
* is_equidistant - Is the time series equidistant in time
* timestep - Time step in seconds (if is_equidistant)
* shape - Shape of each item
* deletevalue - File delete value (NaN value)



## Methods

DataArray has several useful methods for working with data, 
including different ways of *selecting* data:

* [`sel()`](DataArray.sel) - Select subset along an axis
* [`isel()`](DataArray.isel) - Select subset along an axis with an integer

*Aggregations* along an axis:

* [`mean()`](DataArray.mean) - Mean value along an axis
* [`nanmean()`](DataArray.nanmean) - Mean value along an axis (NaN removed)
* [`max()`](DataArray.max) - Max value along an axis
* [`nanmax()`](DataArray.nanmax) - Max value along an axis (NaN removed)
* [`min()`](DataArray.min) - Min value along an axis
* [`nanmin()`](DataArray.nanmin) - Min value along an axis (NaN removed)
* [`aggregate()`](DataArray.aggregate) - Aggregate along an axis
* [`quantile()`](DataArray.quantile) - Quantiles along an axis

*Mathematical operations* +, - and * with numerical values:

* ds + value
* ds - value
* ds * value

and + and - between two DataArrays (if number of items and shapes conform):

* ds1 + ds2
* ds1 - ds2

Other methods that also return a DataArray:

* [`interp_like`](DataArray.interp_like) - Spatio (temporal) interpolation (see [Dfsu interpolation notebook](https://nbviewer.jupyter.org/github/DHI/mikeio/blob/main/notebooks/Dfsu%20-%202D%20interpolation.ipynb))
* [`interp_time()`](DataArray.interp_time) - Temporal interpolation (see [Time interpolation notebook](https://nbviewer.jupyter.org/github/DHI/mikeio/blob/main/notebooks/Time%20interpolation.ipynb))
* [`dropna()`](DataArray.dropna) - Remove time steps where all items are NaN
* [`squeeze()`](DataArray.squeeze) - Remove axes of length 1

*Conversion* methods:

* [`to_xarray()`](DataArray.to_xarray) - Convert DataArray to a xarray DataArray (great for Dfs2)
* [`to_dfs()`](DataArray.to_dfs) - Write DataArray to a Dfs file


## DataArray API

```{eval-rst}
.. autoclass:: mikeio.DataArray
	:members:
```



## DataArray Plotter API

A DataArray `da` can be plotted using `da.plot`. 

```{eval-rst}
.. autoclass:: mikeio.dataarray._DataArrayPlotter
	:members:
```


