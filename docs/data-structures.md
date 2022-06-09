# Data Structures

MIKE IO has these primary data structures: 

* [Dataset](Dataset) - a collection of DataArrays corresponding to the contents of a dfs file; typically obtained from {py:meth}`mikeio.read`
* [DataArray](DataArray) - data and metadata corresponding to one "item" in a dfs file. 
* **Geometry** - spatial description of the data in a dfs file; comes in different flavours: [Grid1D](Grid1D), [Grid2D](Grid2D), [Grid3D](Grid3D), [GeometryFM](GeometryFM), [GeometryFM3D](GeometryFM3D), etc. corresponding to different types of dfs files. 
* **Dfs** - an object returned by `dfs = mikeio.open()` containing the metadata (=header) of a dfs file ready for reading the data (which can be done with `dfs.read()`); exists in different specialized versions: [Dfs0](Dfs0), [Dfs1](Dfs1), [Dfs2](Dfs2), [Dfs3](Dfs3), [Dfsu2DH](Dfsu2DH), [Dfsu3D](Dfsu3D), [Dfsu2DV](Dfsu2DV), [DfsuSpectral](DfsuSpectral), 

