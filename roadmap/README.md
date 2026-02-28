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


- **[Time-Dependent Scaling](features/climate-change-factor.md)** — Apply time-dependent adjustment factors (additive or multiplicative) to DFS file data, e.g. for climate change scenarios or tidal corrections.
- **[Consistent dfs2 and dfsu Plotting](features/consistent-plots.md)** — Unified plotting interface across grid and mesh geometries with consistent styling and options.
- **[Spatial Resampling of dfs2](features/dfs2-resampling.md)** — Resample dfs2 grids to different resolutions or extents using interpolation.
- **[Element Volume in 3D dfsu](features/element-volume-3d.md)** — Compute element volumes for 3D layered meshes, enabling volume-weighted analysis.
- **[Exceedance Statistics](features/exceedance-statistics.md)** — Compute exceedance probabilities, return periods, and threshold-based statistics directly on DataArrays.
- **[Lossless dfsu to NetCDF/Zarr Conversion](features/netcdf-zarr-conversion.md)** — Export dfsu files to NetCDF or Zarr while preserving mesh topology and all metadata.
- **[Horizontal Aggregation in Polygons](features/polygon-aggregation.md)** — Compute area-weighted zonal statistics (mean, sum, min, max) for elements within polygons.
- **[Out-of-Core Temporal Statistics](features/rolling-average-large-files.md)** — Compute time-aggregated and rolling statistics on DFS files too large to fit in memory.

## Not Planned

See [features considered out of scope](features/not-planned.md).
