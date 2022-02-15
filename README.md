
![logo](https://raw.githubusercontent.com/DHI/mikeio/main/images/logo/PNG/MIKE-IO-Logo-Pos-RGB-nomargin.png)
# MIKE IO: input/output of MIKE files in python
 ![Python version](https://img.shields.io/pypi/pyversions/mikeio.svg) 
![Python package](https://github.com/DHI/mikeio/workflows/Python%20package/badge.svg) [![PyPI version](https://badge.fury.io/py/mikeio.svg)](https://badge.fury.io/py/mikeio)


[https://dhi.github.io/mikeio/](https://dhi.github.io/mikeio/)

Read, write and manipulate dfs0, dfs1, dfs2, dfs3, dfsu and mesh files.

Facilitates common data processing workflows for MIKE files.

*For res1d and xns11 files use the related package [MIKE IO 1D](https://github.com/DHI/mikeio1d)*

## Upcoming release: MIKE IO 1.0
MIKE IO 1.0 is planned to be released during the summer of 2022 and it have a lot of benefits, but also require some changes to your code. More details in the [discussion page](https://github.com/DHI/mikeio/discussions/279).

### Important changes
* New class `mikeio.DataArray` with usefule properties and methods
  - item info
  - geometry (grid coordinates)
  - methods for plotting
  - methods for aggreation in time and space
* Indexing into a dataset e.g. `ds["Surface elevation"]` to get a specific item, will not return a numpy array, but a `mikeio.DataArray`

## Requirements
* Windows or Linux operating system
* Python x64 3.7 - 3.10
* (Windows) [VC++ redistributables](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads) (already installed if you have MIKE)

[More info about dependencies](http://docs.mikepoweredbydhi.com/nuget/)

## Where can I get help?

* New ideas and feature requests - [GitHub Discussions](http://github.com/DHI/mikeio/discussions) 
* Bugs - [GitHub Issues](http://github.com/DHI/mikeio/issues) 
* General help, FAQ - [Stackoverflow with the tag `mikeio`](https://stackoverflow.com/questions/tagged/mikeio)

## Installation

From PyPI: 

`pip install mikeio`

Or development version:

`pip install https://github.com/DHI/mikeio/archive/main.zip`


## Examples

### Reading data from dfs0, dfs1, dfs2, dfsu

Generic `read` method to read values, if you need additional features such as coordinates, use specialised classes instead e.g. `Dfsu`

```python
>>> import mikeio
>>> ds = mikeio.read("random.dfs0")
>>> ds
<mikeio.Dataset>
Dimensions: (1000,)
Time: 2017-01-01 00:00:00 - 2017-07-28 03:00:00
Items:
  0:  VarFun01 <Water Level> (meter)
  1:  NotFun <Water Level> (meter)
>>> ds = mikeio.read("random.dfs1")
>>> ds
<mikeio.Dataset>
Dimensions: (100, 3)
Time: 2012-01-01 00:00:00 - 2012-01-01 00:19:48
Items:
  0:  testing water level <Water Level> (meter)
 ```

### Reading dfs0 file into Pandas DataFrame
```python
>>>  from mikeio import Dfs0
>>>  dfs = Dfs0('simple.dfs0')
>>>  ts = dfs.to_dataframe()
```

### Write simple timeseries
```python
>>>  from datetime import datetime
>>>  import numpy as np
>>>  from mikeio import Dfs0
>>>  data = [np.random.random([100])]
>>>  dfs = Dfs0()
>>>  dfs.write('simple.dfs0', data, start_time=datetime(2017, 1, 1), dt=60)

```
### Write timeseries from dataframe
```python
import pandas as pd
import mikeio
>>> df = pd.read_csv(
...         "tests/testdata/co2-mm-mlo.csv",
...         parse_dates=True,
...         index_col="Date",
...         na_values=-99.99,
...     )
>>> df.to_dfs0("mauna_loa.dfs0")
```

For more examples on timeseries data see this [notebook](notebooks/Dfs0%20-%20Timeseries.ipynb)


### Read dfs2 data
```python
>>>  from mikeio import Dfs2
>>> dfs = Dfs2("random.dfs2")
>>> ds = dfs.read()
>>> ds
<mikeio.Dataset>
Dimensions: (3, 100, 2)
Time: 2012-01-01 00:00:00 - 2012-01-01 00:00:24
Items:
  0:  testing water level <Water Level> (meter)
```

### Create dfs2
For a complete example of conversion from netcdf to dfs2 see this [notebook](notebooks/Dfs2%20-%20Sea%20surface%20temperature.ipynb).

Another [example](notebooks/Dfs2%20-%20Global%20Forecasting%20System.ipynb) of downloading meteorological forecast from the Global Forecasting System and converting it to a dfs2 ready to be used by a MIKE 21 model.

### Read dfsu files
```python
>>>  import matplotlib.pyplot as plt
>>>  from mikeio import Dfsu
>>>  dfs = Dfsu("HD.dfsu")
>>>  ds = dfs.read()
>>>  idx = dfs.find_nearest_element(x=608000, y=6907000)
>>>  plt.plot(ds.time, ds.data[0][:,idx])
```
![Timeseries](https://raw.githubusercontent.com/DHI/mikeio/main/images/dfsu_ts.png)

```python
>>>  from mikeio import Mesh
>>>  msh = Mesh("FakeLake.dfsu")
>>>  msh.plot()
```
![Mesh](https://raw.githubusercontent.com/DHI/mikeio/main/images/FakeLake.png)

For more examples on working with dfsu and mesh see these notebooks:
* [Basic dfsu](notebooks/Dfsu%20-%20Read.ipynb)
* [3d dfsu](notebooks/Dfsu%20-%203D%20sigma-z.ipynb)
* [Mesh](notebooks/Mesh.ipynb)
* [Speed & direction](notebooks/Dfsu%20-%20Speed%20and%20direction.ipynb)
* [Dfsu and mesh plotting](notebooks/Dfsu%20and%20Mesh%20-%20Plotting.ipynb)
* [Export to netcdf](notebooks/Dfsu%20-%20Export%20to%20netcdf.ipynb)
* [Export to shapefile](notebooks/Dfsu%20-%20Export%20to%20shapefile.ipynb)


## Pfs

Pfs is the format used for MIKE setup files (.m21fm, .m3fm, .sw etc.).

There is experimental support for reading pfs files, but the API is likely to change.

![pfs](images/pfs.gif)


## Items, units
 Useful when creating a new dfs file
```python
>>> from mikeio.eum import EUMType, EUMUnit
>>> EUMType.Temperature
<EUMType.Temperature: 100006>
>>> EUMType.Temperature.units
[degree Celsius, degree Fahrenheit, degree Kelvin]
>>> EUMUnit.degree_Kelvin
degree Kelvin
```

## Tested

MIKE IO is tested extensively. **95%** total test coverage.

See detailed test coverage report below:
```
File                           Covered  Missed  %
-------------------------------------------------
mikeio\__init__.py               40      1    98%
mikeio\aggregator.py            103      9    91%
mikeio\bin\__init__.py            0      0   100%
mikeio\custom_exceptions.py      19      1    95%
mikeio\dataset.py               272      3    99%
mikeio\dfs.py                   206      7    97%
mikeio\dfs0.py                  239     15    94%
mikeio\dfs1.py                   48      2    96%
mikeio\dfs2.py                  100      2    98%
mikeio\dfs3.py                  201     79    61%
mikeio\dfsu.py                 1337     57    96%
mikeio\dfsutil.py                76      4    95%
mikeio\dotnet.py                 63      4    94%
mikeio\eum.py                  1230      3    99%
mikeio\generic.py               228      2    99%
mikeio\helpers.py                13      0   100%
mikeio\interpolation.py          54      1    98%
mikeio\spatial.py               279      4    99%
mikeio\xyz.py                    12      0   100%
-------------------------------------------------
TOTAL                          5082    229    95%

=================== 335 passed ==================

```

## Cloud enabled

From MIKE IO v.0.7 it is now possible to run MIKE IO on Linux-based Cloud computing, e.g. [Google Colab](https://colab.research.google.com/).

![Colab](images/colab.png)

