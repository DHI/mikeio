---
title: "Zarr Export"
status: "Under Consideration"
category: "Interoperability"
summary: "Export DFS data to Zarr stores for cloud-native access, parallel reads, and integration with cloud-hosted analysis workflows."
---

## Value Proposition

Zarr is the cloud-native array storage format adopted by the Pangeo community and supported natively by xarray, Dask, and major cloud platforms. Converting MIKE simulation results to Zarr enables efficient access from anywhere — no file downloads, no proprietary readers, just standard HTTP range requests.

## What This Enables

- **Cloud storage**: Push simulation results to S3/GCS/Azure Blob as Zarr stores, accessible from any machine
- **Parallel reads**: Multiple workers read different chunks concurrently without file locking
- **Incremental writes**: Append new timesteps to an existing Zarr store as a simulation progresses
- **Web visualization**: Zarr stores can back web-based dashboards (e.g., via xarray + hvplot/Panel)
- **Large dataset handling**: Zarr's chunking eliminates the need to load entire files into memory

## Current State

No Zarr export exists in MIKE IO. The `netcdf-zarr-conversion` roadmap item identifies the need. With xarray's `to_zarr()`, the technical path is straightforward once UGRID-compliant xarray conversion is in place — the primary challenge is preserving mesh topology metadata.

## Design Considerations

- Build on top of UGRID-compliant xarray export — `ds.to_xarray().to_zarr()` should produce a valid, self-describing Zarr store
- Target Zarr v3 with sharding codec to reduce object count for cloud storage (many small chunks → fewer sharded objects)
- Chunking strategy matters: time-chunked (one chunk per timestep) suits temporal analysis; spatial chunking is harder for unstructured meshes but relevant for structured grids (dfs2, dfs3)
- Include consolidated metadata for fast opening (single metadata fetch instead of one per variable)
- Consider a direct `ds.to_zarr()` method on MIKE IO Dataset as a convenience wrapper
- Mesh topology variables (node coordinates, connectivity) should be written as unchunked arrays
