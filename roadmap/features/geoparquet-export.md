---
title: "GeoParquet Export"
status: "Under Consideration"
category: "Interoperability"
summary: "Export mesh element results as GeoParquet files for cloud-friendly columnar storage, fast spatial queries, and interoperability with modern data tools."
---

## Value Proposition

GeoParquet is an emerging standard for geospatial vector data in columnar format. It combines Parquet's strengths (compression, predicate pushdown, columnar access) with embedded geometry and CRS metadata. For MIKE IO users, this means simulation results can be stored, shared, and queried using tools like DuckDB, Polars, pandas, and cloud data platforms — without any GIS software.

## What This Enables

- **SQL queries on results**: `SELECT * FROM 'results.parquet' WHERE ST_Within(geometry, ?)` via DuckDB spatial
- **Cloud-native sharing**: GeoParquet files on S3 are directly queryable with row-group filtering — no full download needed
- **Dashboard backends**: Parquet files are fast data sources for dashboards and web applications
- **Cross-tool compatibility**: Readable by QGIS, GeoPandas, Polars, DuckDB, BigQuery, Snowflake, and others
- **Time series at elements**: Store all timesteps for selected elements as a tall table with element geometry, time, and values

## Current State

No Parquet or GeoParquet export exists. Building on `to_geodataframe()` (see GeoDataFrame Export feature), GeoParquet export would be a thin wrapper: `gdf.to_parquet("results.parquet")`.

## Design Considerations

- Depends on the `to_geodataframe()` feature for geometry construction
- Default to one row per element per timestep (tall/tidy format) for time-varying data
- For single-timestep snapshots, one row per element with all variables as columns (wide format)
- Include CRS metadata in the Parquet file per the GeoParquet spec (PROJJSON in column metadata)
- Element area could be included as a column to support area-weighted aggregation downstream
- Consider a `to_parquet()` convenience method directly on DataArray/Dataset that handles the GeoDataFrame conversion internally
- pyarrow and geopandas should remain optional dependencies
