# Dataset

The [Dataset](#mikeio.Dataset>) is the common MIKE IO data structure 
for data from dfs files. 
The `mikeio.read()` methods returns a Dataset as a container of DataArrays (Dfs items)

Each DataArray have the following properties:
* **item** - an  `ItemInfo <#mikeio.eum.ItemInfo>` with name, type and unit
* **time** - a pandas.DateTimeIndex with the time instances of the data
* **values** - a NumPy array

Use Dataset's string representation to get an overview of the Dataset


```python
>>> import mikeio
>>> ds = mikeio.read("testdata/HD2D.dfsu")
>>> ds
<mikeio.Dataset>
Geometry: Dfsu2D
Dimensions: (time:9, element:884)
Time: 1985-08-06 07:00:00 - 1985-08-07 03:00:00 (9 records)
Items:
  0:  Surface elevation <Surface Elevation> (meter)        
  1:  U velocity <u velocity component> (meter per sec)    
  2:  V velocity <v velocity component> (meter per sec)    
  3:  Current speed <Current Speed> (meter per sec)   
```

Selecting items
---------------
Selecting a specific item "itemA" (at position 0) from a Dataset ds can be 
done with:

* ds[["itemA"]] - returns a new Dataset with "itemA"
* ds["itemA"] - returns the data of "itemA"
* ds[[0]] - returns a new Dataset with "itemA" 
* ds[0] - returns the data of "itemA"

```
>>> ds.Surface_elevation
<mikeio.DataArray>
Name: Surface elevation
Geometry: Dfsu2D
Dimensions: (time:9, element:884)
Time: 1985-08-06 07:00:00 - 1985-08-07 03:00:00 (9 records)
```

Negative index e.g. ds[-1] can also be used to select from the end. 
Several items ("itemA" at 0 and "itemC" at 2) can be selected with the notation:

* ds[["itemA", "itemC"]]
* ds[[0, 2]]

Note that this behavior is similar to pandas and xarray.


Selecting timesteps or elements
-------------------------------
The `isel() <#mikeio.Dataset.isel>`_ method can be used for selecting specific timesteps or elements across a Dataset. 

* ds.isel([0, 1], axis=0) - selects timestep 0 and 1
* ds.isel([3,78], axis=1) - selects element 3 and 78

A date range can also be selected using the slice notation (similar to pandas DataFrame): 

* ds[start_time:end_time] - selects all time steps between start and end (either can be empty)

```python
>>> ds["1985-8-7":]
<mikeio.Dataset>
Geometry: Dfsu2D
Dimensions: (time:2, element:884)
Time: 1985-08-07 00:30:00 - 1985-08-07 03:00:00 (2 records)
Items:
  0:  Surface elevation <Surface Elevation> (meter)
  1:  U velocity <u velocity component> (meter per sec)
  2:  V velocity <v velocity component> (meter per sec)
  3:  Current speed <Current Speed> (meter per sec)
```

Properties
----------
The Dataset has several convenience properties 
(besides the main properties items, time and data):

* n_items - Number of items
* n_timesteps - Number of timesteps
* n_elements - Number of elements
* start_time - First time instance (as datetime)
* end_time - Last time instance (as datetime)
* is_equidistant - Is the time series equidistant in time
* timestep - Time step in seconds (if is_equidistant)
* shape - Shape of each item
* deletevalue - File delete value (NaN value)



Methods
-------
Dataset has several useful methods for working with data, 
including different ways of *selecting* data:

* `head() <#mikeio.Dataset.head>`_ - Return the first n timesteps
* `tail() <#mikeio.Dataset.tail>`_ - Return the last n timesteps
* `thin() <#mikeio.Dataset.thin>`_ - Return every n:th timesteps
* `isel() <#mikeio.Dataset.isel>`_ - Select subset along an axis

*Aggregations* along an axis:

* `mean() <#mikeio.Dataset.mean>`_ - Mean value along an axis
* `nanmean() <#mikeio.Dataset.nanmean>`_ - Mean value along an axis (NaN removed)
* `max() <#mikeio.Dataset.max>`_ - Max value along an axis
* `nanmax() <#mikeio.Dataset.nanmax>`_ - Max value along an axis (NaN removed)
* `min() <#mikeio.Dataset.min>`_ - Min value along an axis
* `nanmin() <#mikeio.Dataset.nanmin>`_ - Min value along an axis (NaN removed)
* `average() <#mikeio.Dataset.average>`_ - Compute the weighted average along the specified axis.
* `aggregate() <#mikeio.Dataset.aggregate>`_ - Aggregate along an axis
* `quantile() <#mikeio.Dataset.quantile>`_ - Quantiles along an axis
* `nanquantile() <#mikeio.Dataset.nanquantile>`_ - Quantiles along an axis (NaN ignored)

*Mathematical operations* +, - and * with numerical values:

* ds + value
* ds - value
* ds * value

and + and - between two Datasets (if number of items and shapes conform):

* ds1 + ds2
* ds1 - ds2

Other methods that also return a Dataset:

* `interp_time() <#mikeio.Dataset.interp_time>`_ - Temporal interpolation (see `Time interpolation notebook <https://nbviewer.jupyter.org/github/DHI/mikeio/blob/main/notebooks/Time%20interpolation.ipynb>`_)
* `dropna() <#mikeio.Dataset.dropna>`_ - Remove time steps where all items are NaN
* `squeeze() <#mikeio.Dataset.squeeze>`_ - Remove axes of length 1

*Conversion* methods:

* `to_dataframe() <#mikeio.Dataset.to_dataframe>`_ - Convert Dataset to a Pandas DataFrame
* `to_xarray() <#mikeio.Dataset.to_xarray>`_ - Convert Dataset to a xarray dataset (great for Dfs2)
* `to_dfs() <#mikeio.Dataset.to_dfs>`_ - Write Dataset to a Dfs file



Dataset API
-----------
```{eval-rst}
.. autoclass:: mikeio.Dataset
	:members:
```

DataArray API
-----------
```{eval-rst}
.. autoclass:: mikeio.DataArray
	:members:
```