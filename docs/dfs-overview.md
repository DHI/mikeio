# Dfs Overview

MIKE IO has a similar API for the three gridded dfs file types: [Dfs1](Dfs1), [Dfs2](Dfs2) and [Dfs3](Dfs3). 

All Dfs classes (and the Dataset) class are representations of timeseries and 
share these properties: 

* items - a list of [ItemInfo](ItemInfo) with name, type and unit of each item
* n_items - Number of items
* n_timesteps - Number of timesteps
* start_time - First time instance (as datetime)
* end_time - Last time instance (as datetime)
* geometry - spatial description of the data in the file ([Grid1D](Grid1D), [Grid2D](Grid2D), or [Grid3D](Grid3D))
* deletevalue - File delete value (NaN value)



## Open or read? 

Dfs files contain data and meta data. 

If the file is small (e.g. <100 MB), you probably just want to get all the data at once with `mikeio.read(...)` which will return a `Dataset` for further processing.   

If the file is big, you will typically get the file *header* with `dfs = mikeio.open(...)` which will return a MIKE IO Dfs class, before reading any data. When you have decided what to read (e.g. specific time steps, an sub area or selected elements), you can the get the Dataset `ds` you need with `ds = dfs.read(...)`.
