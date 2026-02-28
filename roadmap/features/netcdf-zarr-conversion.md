---
title: "Lossless dfsu to NetCDF/Zarr Conversion"
status: "Under Consideration"
category: "Interoperability"
summary: "Export dfsu files to NetCDF or Zarr while preserving mesh topology and all metadata."
---

## Value Proposition

NetCDF and Zarr are widely supported by the scientific Python ecosystem (xarray, Dask, cloud storage). Converting dfsu files to these formats would unlock lazy loading, parallel computation, and cloud-native workflows — without losing the mesh structure or metadata that makes the data meaningful.

## What This Enables

- **Cloud-native workflows**: Store simulation results in Zarr for efficient cloud access
- **Lazy loading with xarray**: Process datasets larger than memory using Dask-backed arrays
- **Interoperability**: Share results with collaborators who don't have MIKE IO installed
- **Round-trip fidelity**: Convert to NetCDF/Zarr and back without losing mesh topology, projection, or EUM metadata

## Current State

`to_xarray()` exists on Dataset and DataArray, enabling conversion to xarray objects. However, there is no direct export path that preserves mesh topology in a standardised format (e.g., UGRID conventions for NetCDF).
