# pydhi
Facilitate creating, reading and writing dfs0, dfs2, dfs1 and dfs3 files. Reading Res1D data.

# Examples

## Reading dfs0 file into Pandas DataFrame
```python
from pydhi import dfs0 as dfs0
dfs = dfs0.dfs0()
ts = dfs.read_to_pandas(dfs0file)
```

## Create simple timeseries
```python
import datetime
import numpy as np
from pydhi import dfs0 as dfs0

data = []
nt = 100
d = np.random.random([nt])
start_time = datetime.datetime(2017, 1, 1)
dt = 60 # using default timestep_unit of second
data.append(d)
dfs = dfs0.dfs0()
dfs.create(filename='simple.dfs0', data=data,
           start_time=start_time,dt=dt )

```


## Create non-equidistant dfs0
```python
d1 = np.random.random([1000])
d2 = np.random.random([1000])
data = []
data.append(d1)
data.append(d2)
start_time = datetime.datetime(2017, 1, 1)
timeseries_unit = 1402
title = 'Hello Test'
names = ['VarFun01', 'NotFun']
variable_type = [100000, 100000]
unit = [1000, 1000]
data_value_type = [0, 1]
dt = 5
dfs = dfs0.dfs0()
dfs.create(filename='random.dfs0', data=data,
        	start_time=start_time,
            timeseries_unit=timeseries_unit, dt=dt,
            names=names, title=title,
            variable_type=variable_type,
            unit=unit, data_value_type=data_value_type)

```

## Create non equidistant dfs0
```python
d1 = np.random.random([1000])
d2 = np.random.random([1000])
data = []
data.append(d1)
data.append(d2)
start_time = datetime.datetime(2017, 1, 1)
time_vector = []
for i in range(1000):
	time_vector.append(start_time + datetime.timedelta(hours=i*0.1))
title = 'Hello Test'
names = ['VarFun01', 'NotFun']
variable_type = [100000, 100000]
unit = [1000, 1000]
data_value_type = [0, 1]

dfs = dfs0.dfs0()
dfs.create(filename='neq.dfs0', data=data,
			datetimes=time_vector,
			names=names, title=title,
			variable_type=variable_type, unit=unit,
			data_value_type=data_value_type)
```

## Read dfs2 data
```python
dfs2File = r"C:\test\random.dfs2"
dfs = dfs2.dfs2()
data = dfs.read(dfs2File, [0])[0]
data = data[0]
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
import pydhi

dfs = pydhi.dfsu.dfsu()

filename = "HD.dfsu"
(d,t,n)= dfs.read(filename,[0])

idx = dfs.find_closest_element_index(x=608000, y=6907000)

plt.plot(t,d[0][idx,:])
```
![Timeseries](images/dfsu_ts.png)


