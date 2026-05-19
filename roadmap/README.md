# MIKE IO Product Roadmap

This roadmap outlines the current and future direction of MIKE IO — a Python package for reading, writing, and manipulating MIKE files (dfs0, dfs1, dfs2, dfs3, dfsu, mesh).

For questions or feature requests, please open a [GitHub Discussion](https://github.com/DHI/mikeio/discussions).

---


## Delivered


- **[Cross-Platform Support](features/cross-platform.md)** — Native Python DFS I/O via mikecore, replacing the Windows-only pythonnet dependency.
- **[Flexible Mesh (dfsu) Support](features/flexible-mesh.md)** — Full read/write support for 2D and 3D unstructured mesh files, including spectral data.
- **[PFS File Support](features/pfs-support.md)** — Read, modify, and write MIKE parameter files (.pfs) as structured Python objects.
- **[Plotting](features/plotting.md)** — Geometry-aware matplotlib plotting for time series, grids, and flexible meshes.

## Under Consideration


- **[EUM to CF Standard Name Mapping](features/cf-standard-name-mapping.md)** — Map DHI EUM types and units to CF convention standard names and UDUNITS, enabling standards-compliant metadata in NetCDF, Zarr, and xarray exports.
- **[Time-Dependent Scaling](features/climate-change-factor.md)** — Apply time-dependent adjustment factors (additive or multiplicative) to DFS file data, e.g. for climate change scenarios or tidal corrections.
- **[Consistent dfs2 and dfsu Plotting](features/consistent-plots.md)** — Unified plotting interface across grid and mesh geometries with consistent styling and options.
- **[Spatial Resampling of dfs2](features/dfs2-resampling.md)** — Resample dfs2 grids to different resolutions or extents using interpolation.
- **[Element Volume in 3D dfsu](features/element-volume-3d.md)** — Compute element volumes for 3D layered meshes, enabling volume-weighted analysis.
- **[Exceedance Statistics](features/exceedance-statistics.md)** — Compute exceedance probabilities, return periods, and threshold-based statistics directly on DataArrays.
- **[GeoParquet Export](features/geoparquet-export.md)** — Export mesh element results as GeoParquet files for cloud-friendly columnar storage, fast spatial queries, and interoperability with modern data tools.
- **[Horizontal Aggregation in Polygons](features/polygon-aggregation.md)** — Compute area-weighted zonal statistics (mean, sum, min, max) for elements within polygons.
- **[Out-of-Core Temporal Statistics](features/rolling-average-large-files.md)** — Compute time-aggregated and rolling statistics on DFS files too large to fit in memory.
- **[GeoDataFrame Export](features/to-geodataframe.md)** — Export DataArray and Dataset contents as GeoDataFrames with polygon or point geometries and CRS metadata, enabling direct use with GIS tools.
- **[UGRID-Compliant NetCDF Export](features/ugrid-netcdf-export.md)** — Export dfsu mesh topology and data as UGRID-compliant NetCDF, readable by QGIS, ParaView, xugrid, and other standards-aware tools.
- **[xarray Backend Entry Point](features/xarray-backend.md)** — Register MIKE IO as an xarray backend engine, enabling lazy loading of DFS files via xr.open_dataset(engine='mikeio').
- **[Zarr Export](features/zarr-export.md)** — Export DFS data to Zarr stores for cloud-native access, parallel reads, and integration with cloud-hosted analysis workflows.

## Not Planned

See [features considered out of scope](features/not-planned.md).
