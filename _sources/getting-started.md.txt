# Getting started

## Resources

* Online book: [Getting started with Dfs files in Python using MIKE IO](https://dhi.github.io/getting-started-with-mikeio)
* Online book: [Python for marine modelers using MIKE IO and FMskill](https://dhi.github.io/book-learn-mikeio-fmskill)
* [Example notebooks](https://nbviewer.jupyter.org/github/DHI/mikeio/tree/main/notebooks/)
* [Unit tests](https://github.com/DHI/mikeio/tree/main/tests)
* [DFS file system specification](https://docs.mikepoweredbydhi.com/core_libraries/dfs/dfs-file-system)


## Dataset
The [Dataset](Dataset) is the common MIKE IO data structure for data read from dfs files. 
The  `mikeio.read()` method returns a Dataset with a [DataArray](DataArray) for each item.

The DataArray have all the relevant information, e.g:

* item - an [ItemInfo](eum.ItemInfo) with name, type and unit
* time - a pandas.DateTimeIndex with the time instances of the data
* values - a NumPy array

## Items, ItemInfo and EUM

The dfs items in MIKE IO are represented by the [ItemInfo class](eum.ItemInfo).
An ItemInfo consists of:

* name - a user-defined string 
* type - an [EUMType](eum.EUMType) 
* unit - an [EUMUnit](eum.EUMUnit)

```python
>>> from mikeio import ItemInfo, EUMType
>>> item = ItemInfo("Viken", EUMType.Water_Level)
>>> item
Viken <Water Level> (meter)
>>> ItemInfo(EUMType.Wind_speed)
Wind speed <Wind speed> (meter per sec)
```

## Dfs0
A dfs0 file is also called a time series file. 

Read Dfs0 to Dataset:

```python
>>> ds = mikeio.read("testdata/da_diagnostic.dfs0")
>>> ds
<mikeio.Dataset>
dims: (time:744)
time: 2017-10-27 00:00:00 - 2017-10-29 18:00:00 (744 non-equidistant records)
geometry: GeometryUndefined()
items:
  0:  State 1Sign. Wave Height <Significant wave height> (meter)
  1:  State 2Sign. Wave Height <Significant wave height> (meter)
  2:  Mean StateSign. Wave Height <Significant wave height> (meter)
  3:  MeasurementSign. Wave Height <Significant wave height> (meter)
```

Read more on the [Dfs0 page](dfs0.md).

Convert the timeseries dataset to a pandas DataFrame:

```python
>>> df = ds.to_dataframe()
```


## Dfs2

A dfs2 file is also called a grid series file. Values in a dfs2 file are ‘element based’, i.e. values are defined in the centre of each grid cell. 

```python
>>> ds = mikeio.read("gebco_sound.dfs2") 
>>> ds
<mikeio.Dataset>
Dimensions: (time:1, y:264, x:216)
Time: 2020-05-15 11:04:52 (time-invariant)
Items:
  0:  Elevation <Total Water Depth> (meter)
```

Read more on the [Dfs2 page](dfs2.md).


## Generic dfs
MIKE IO has [`generic`](generic.md) functionality that works for all dfs files: 

* [`concat()`](generic.concat) - Concatenates files along the time axis
* [`extract()`](generic.extract) - Extract timesteps and/or items to a new dfs file
* [`diff()`](generic.diff) - Calculate difference between two dfs files with identical geometry
* [`sum()`](generic.sum) - Calculate the sum of two dfs files
* [`scale()`](generic.scale) - Apply scaling to any dfs file
* [`avg_time()`](generic.avg_time) - Create a temporally averaged dfs file
* [`quantile()`](generic.quantile) - Create a dfs file with temporal quantiles

All generic methods creates a new dfs file.

```python
from mikeio import generic
generic.concat(["fileA.dfs2", "fileB.dfs2"], "new_file.dfs2")
```

See [Generic page](generic.md) and the [Generic notebook](<https://nbviewer.jupyter.org/github/DHI/mikeio/blob/main/notebooks/Generic.ipynb>) for more examples.
