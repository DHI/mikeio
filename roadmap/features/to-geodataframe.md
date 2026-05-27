---
title: "GeoDataFrame Export"
status: "Under Consideration"
category: "Interoperability"
summary: "Export DataArray and Dataset contents as GeoDataFrames with polygon or point geometries and CRS metadata, enabling direct use with GIS tools."
---

## Value Proposition

GeoDataFrame is the standard interchange format for vector geospatial data in Python. A `to_geodataframe()` method on DataArray would let users go from dfsu results to shapefile, GeoPackage, GeoJSON, or GeoParquet in one step — and access the full geopandas/shapely ecosystem for spatial analysis, joins, and visualization.

## What This Enables

- **GIS export**: Write dfsu results to GeoPackage, shapefile, or GeoJSON via `gdf.to_file()`
- **Spatial joins**: Intersect MIKE results with other geospatial datasets (land use, catchments, monitoring stations)
- **Interactive maps**: `gdf.explore()` renders results on an interactive Leaflet map in Jupyter/marimo
- **GeoParquet output**: `gdf.to_parquet()` for cloud-friendly columnar storage (see GeoParquet feature)
- **DuckDB/Polars spatial queries**: GeoParquet files are queryable by DuckDB's spatial extension

## Current State

`GeometryFM2D.to_shapely()` returns a `MultiPolygon` of mesh elements. CRS metadata is available via `spatial/crs.py`. The building blocks exist, but there is no method that combines geometry, data values, and CRS into a GeoDataFrame.

## Design Considerations

- For dfsu (unstructured mesh): each row is a mesh element with a polygon geometry and variable values at a single timestep
- For dfs2 (structured grid): each row is a grid cell with a polygon geometry
- For dfs0 (time series at a point): each row is a timestep with a point geometry
- Multi-timestep data should default to a single timestep; a `timestep` parameter selects which one
- CRS should be set from the file's projection metadata via `to_pyproj()`
- geopandas should remain an optional dependency — import lazily and raise a helpful error if not installed
