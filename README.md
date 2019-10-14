# pydhi
Facilitate creating, reading and writing dfs0, dfs2, dfs1 and dfs3 files. Reading Res1D data.

# Examples

## Reading dfs0 file into Pandas DataFrame
```python
from pydhi.dfs0 import dfs0
dfs = dfs0()
ts = dfs.read_to_pandas('simple.dfs0')
```

## Create simple timeseries
```python
from datetime import datetime, timedelta
import numpy as np
from pydhi.dfs0 import dfs0

data = []
nt = 100
d = np.random.random([nt])
start_time = datetime(2017, 1, 1)
dt = 60 # using default timestep_unit of second
data.append(d)
dfs = dfs0()
dfs.create(filename='simple.dfs0', data=data,
           start_time=start_time,dt=dt )

```


## Create equidistant dfs0 with daily timestep
```python
from pydhi.eum import TimeStep
d1 = np.random.random([1000])
d2 = np.random.random([1000])
data = []
data.append(d1)
data.append(d2)
start_time = datetime(2017, 1, 1)
timeseries_unit = TimeStep.DAY
title = 'Hello Test'
names = ['VarFun01', 'NotFun']
dt = 5
dfs = dfs0.dfs0()
dfs.create(filename='random.dfs0', data=data,
            start_time=start_time,
            timeseries_unit=timeseries_unit, dt=dt,
            names=names, title=title,
            variable_type=variable_type,
            unit=unit, data_value_type=data_value_type)

```
For more examples see this [notebook](notebooks/01%20-%20Timeseries.ipynb)


## Read dfs2 data
```python
from pydhi.dfs2 import dfs2
dfs2File = r"20150101-DMI-L4UHfnd-NSEABALTIC-v01-fv01-DMI_OI.dfs2"
dfs = dfs2()
res = dfs.read(dfs2File)
res.names
```

## Create dfs2
For a complete example of conversion from netcdf to dfs2 see this [notebook](notebooks/Sea%20surface%20temperature%20-%20dfs2.ipynb)

## DFS Utilities to query variable type, time series types (useful when creating a new dfs file)
```python
>>> from pydhi.dfs_util import type_list, unit_list
>>> type_list('Water level')
{100000: 'Water Level', 100307: 'Water level change'}

>>> unit_list(100307)
{1000: 'meter', 1003: 'feet'}
```

## Read Res1D file Return Pandas DataFrame
```python
import res1d as r1d
p1 = r1d.ExtractionPoint()
p1.BranchName  = 'branch1'
p1.Chainage = 10.11
p1.VariableType  = 'Discharge'
ts = r1d.read('res1dfile.res1d', [p1])
```

## Read dfsu files
```python
import matplotlib.pyplot as plt
from pydhi.dfsu import dfsu

dfs = dfsu()

filename = "HD.dfsu"
res = dfs.read(filename)

idx = dfs.find_closest_element_index(x=608000, y=6907000)

# data has two dimensions time, x
plt.plot(res.time, res.data[0][:,idx])
```
![Timeseries](images/dfsu_ts.png)


