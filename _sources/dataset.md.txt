# Dataset

The [Dataset](Dataset) is the MIKE IO data structure 
for data from dfs files. 
The {py:meth}`mikeio.read` methods returns a Dataset as a container of [DataArrays](dataarray) (Dfs items). Each DataArray has the properties, *item*, *time*, *geometry* and *values*. The time and geometry are common to all DataArrays in the Dataset. 

The Dataset has the following primary properties: 

* **items** - a list of {py:class}`mikeio.eum.ItemInfo` items for each dataarray
* **time** - a {py:class}`pandas.DatetimeIndex` with the time instances of the data
* **geometry** - a Geometry object with the spatial description of the data


Use Dataset's string representation to get an overview of the Dataset


```python
>>> import mikeio
>>> ds = mikeio.read("testdata/HD2D.dfsu")
>>> ds
<mikeio.Dataset>
dims: (time:9, element:884)
time: 1985-08-06 07:00:00 - 1985-08-07 03:00:00 (9 records)
geometry: Dfsu2D (884 elements, 529 nodes)
items:
  0:  Surface elevation <Surface Elevation> (meter)
  1:  U velocity <u velocity component> (meter per sec)
  2:  V velocity <v velocity component> (meter per sec)
  3:  Current speed <Current Speed> (meter per sec)
```

## Selecting items

Selecting a specific item "itemA" (at position 0) from a Dataset ds can be done with:

* `ds[["itemA"]]` - returns a new Dataset with "itemA"
* `ds["itemA"]` - returns "itemA" DataArray
* `ds[[0]]` - returns a new Dataset with "itemA" 
* `ds[0]` - returns "itemA" DataArray
* `ds.itemA` - returns "itemA" DataArray

We recommend the use *named* items for readability. 

```
>>> ds.Surface_elevation
<mikeio.DataArray>
name: Surface elevation
dims: (time:9, element:884)
time: 1985-08-06 07:00:00 - 1985-08-07 03:00:00 (9 records)
geometry: Dfsu2D (884 elements, 529 nodes)
```

Negative index e.g. ds[-1] can also be used to select from the end. 
Several items ("itemA" at 0 and "itemC" at 2) can be selected with the notation:

* ds[["itemA", "itemC"]]
* ds[[0, 2]]

Note that this behavior is similar to pandas and xarray.


## Temporal selection

A time slice of a Dataset can be selected in several different ways. 

```python
>>> ds.sel(time="1985-08-06 12:00")
<mikeio.Dataset>
dims: (element:884)
time: 1985-08-06 12:00:00 (time-invariant)
geometry: Dfsu2D (884 elements, 529 nodes)
items:
  0:  Surface elevation <Surface Elevation> (meter)
  1:  U velocity <u velocity component> (meter per sec)
  2:  V velocity <v velocity component> (meter per sec)
  3:  Current speed <Current Speed> (meter per sec)

>>> ds["1985-8-7":]
<mikeio.Dataset>
dims: (time:2, element:884)
time: 1985-08-07 00:30:00 - 1985-08-07 03:00:00 (2 records)
geometry: Dfsu2D (884 elements, 529 nodes)
items:
  0:  Surface elevation <Surface Elevation> (meter)
  1:  U velocity <u velocity component> (meter per sec)
  2:  V velocity <v velocity component> (meter per sec)
  3:  Current speed <Current Speed> (meter per sec)

```

## Spatial selection

The `sel` method finds the nearest element.

```python
>>> ds.sel(x=607002, y=6906734)
<mikeio.Dataset>
dims: (time:9)
time: 1985-08-06 07:00:00 - 1985-08-07 03:00:00 (9 records)
geometry: GeometryPoint2D(x=607002.7094112666, y=6906734.833048992)
items:
  0:  Surface elevation <Surface Elevation> (meter)
  1:  U velocity <u velocity component> (meter per sec)
  2:  V velocity <v velocity component> (meter per sec)
  3:  Current speed <Current Speed> (meter per sec)
```


## Plotting

In most cases, you will *not* plot the Dataset, but rather it's DataArrays. But there are two exceptions: 

* dfs0-Dataset : plot all items as timeseries with ds.plot()
* scatter : compare two items using ds.plot.scatter(x="itemA", y="itemB")

See details in the [API specification](_DatasetPlotter) below.


## Properties
The Dataset (and DataArray) has several properties:

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

Dataset (and DataArray) has several useful methods for working with data, 
including different ways of *selecting* data:

* [`sel()`](Dataset.sel) - Select subset along an axis
* [`isel()`](Dataset.isel) - Select subset along an axis with an integer

*Aggregations* along an axis:

* [`mean()`](Dataset.mean) - Mean value along an axis
* [`nanmean()`](Dataset.nanmean) - Mean value along an axis (NaN removed)
* [`max()`](Dataset.max) - Max value along an axis
* [`nanmax()`](Dataset.nanmax) - Max value along an axis (NaN removed)
* [`min()`](Dataset.min) - Min value along an axis
* [`nanmin()`](Dataset.nanmin) - Min value along an axis (NaN removed)
* [`average()`](Dataset.average) - Compute the weighted average along the specified axis.
* [`aggregate()`](Dataset.aggregate) - Aggregate along an axis
* [`quantile()`](Dataset.quantile) - Quantiles along an axis
* [`nanquantile()`](Dataset.nanquantile) - Quantiles along an axis (NaN ignored)

*Mathematical operations* +, - and * with numerical values:

* ds + value
* ds - value
* ds * value

and + and - between two Datasets (if number of items and shapes conform):

* ds1 + ds2
* ds1 - ds2

Other methods that also return a Dataset:

* [`interp_like`](Dataset.interp_like) - Spatio (temporal) interpolation (see [Dfsu interpolation notebook](https://nbviewer.jupyter.org/github/DHI/mikeio/blob/main/notebooks/Dfsu%20-%202D%20interpolation.ipynb))
* [`interp_time()`](Dataset.interp_time) - Temporal interpolation (see [Time interpolation notebook](https://nbviewer.jupyter.org/github/DHI/mikeio/blob/main/notebooks/Time%20interpolation.ipynb))
* [`dropna()`](Dataset.dropna) - Remove time steps where all items are NaN
* [`squeeze()`](Dataset.squeeze) - Remove axes of length 1

*Conversion* methods:

* [`to_dataframe()`](Dataset.to_dataframe) - Convert Dataset to a {py:class}`pandas.DataFrame`
* [`to_xarray()`](Dataset.to_xarray) - Convert Dataset to a {py:class}`xarray.Dataset` (great for Dfs2)
* [`to_dfs()`](Dataset.to_dfs) - Write Dataset to a Dfs file



## Dataset API

```{eval-rst}
.. autoclass:: mikeio.Dataset
	:members:
```


## Dataset Plotter API

```{eval-rst}
.. autoclass:: mikeio.dataset._DatasetPlotter
	:members:
```


