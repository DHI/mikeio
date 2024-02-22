# Data Structures

MIKE IO has these primary data structures: 

* [**Dataset**](dataset.md) - a collection of DataArrays corresponding to the contents of a dfs file; typically obtained from {py:meth}`mikeio.read`
* [**DataArray**](dataarray.md) - data and metadata corresponding to one "item" in a dfs file. 
* **Geometry** - spatial description of the data in a dfs file; comes in different flavours: [Grid1D](`mikeio.Grid1D`), [Grid2D](`mikeio.Grid2D`), [Grid3D](`mikeio.Grid3D`), [GeometryFM](`mikeio.GeometryFM`), [GeometryFM3D](`mikeio.GeometryFM3D`), etc. corresponding to different types of dfs files. 
* **Dfs** - an object returned by `dfs = mikeio.open()` containing the metadata (=header) of a dfs file ready for reading the data (which can be done with `dfs.read()`); exists in different specialized versions: [Dfs0](`mikeio.Dfs0`), [Dfs1](`mikeio.Dfs1`), [Dfs2](`mikeio.Dfs2`), [Dfs3](`mikeio.Dfs3`), [Dfsu2DH](`mikeio.Dfsu2DH`), [Dfsu3D](`mikeio.Dfsu3D`), [Dfsu2DV](`mikeio.Dfsu2DV`), [DfsuSpectral](`mikeio.DfsuSpectral`).

