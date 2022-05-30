# Data Structures

MIKE IO has these primary data structures: 

* [Dataset](Dataset) - a collection of DataArrays corresponding to the contents of a dfs file; typically obtained from `mikeio.read()`
* [DataArray](DataArray) - data and meta data corresponding to one "item" in a dfs file. 
* **Geometry** - spatial description of the data in a dfs file; comes in different flavours: [Grid1D](Grid1D), [Grid2D](Grid2D), [Grid3D](Grid3D), [GeometryFM](GeometryFM), [GeometryFM3D](GeometryFM3D), etc. corresponding to different types of dfs files. 
* **Dfs** - an object returned by `mikeio.read()` containing the meta data (=header) of a dfs file ready for reading the data; exists in different specialized versions, e.g. [Dfs1](dfs123), [Dfs2](Dfs2), [Dfs3](dfs123), [Dfsu](dfsu) 

