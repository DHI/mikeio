---
title: "xarray Backend Entry Point"
status: "Under Consideration"
category: "Interoperability"
summary: "Register MIKE IO as an xarray backend engine, enabling lazy loading of DFS files via xr.open_dataset(engine='mikeio')."
---

## Value Proposition

xarray's backend entry point system allows any file format to be opened with `xr.open_dataset()`. Registering MIKE IO as a backend would make DFS files first-class citizens in the xarray ecosystem — compatible with Dask for lazy/parallel computation, xarray's selection and aggregation API, and downstream tools like hvplot, cf-xarray, and flox.

## What This Enables

- **Lazy loading**: `xr.open_dataset("large.dfsu", engine="mikeio")` returns a Dataset without reading all data into memory
- **Dask integration**: `xr.open_dataset(..., chunks={"time": 10})` enables out-of-core computation on large files
- **Multi-file datasets**: `xr.open_mfdataset(["run1.dfsu", "run2.dfsu"], engine="mikeio")` for concatenating results
- **Ecosystem compatibility**: Any tool that accepts xarray Datasets works with MIKE files immediately
- **Notebook workflows**: Users familiar with xarray can work with MIKE files without learning a separate API

## Current State

`to_xarray()` on Dataset and DataArray performs an eager conversion — all data is read into memory first, then wrapped as xarray objects. There is no lazy path. The Rust DFS engine experiment (branch `experiment/thinking`) has a working `BackendEntrypoint` with `DfsBackendArray` for lazy timestep reads, demonstrating feasibility.

## Design Considerations

- Implement `xarray.backends.BackendEntrypoint` subclass with `open_dataset()` and `guess_can_open()` methods
- Register via `[project.entry-points."xarray.backends"]` in `pyproject.toml`
- Data variables should be backed by a lazy array wrapper that reads individual timesteps on demand
- Coordinates should include time, and spatial coordinates appropriate to the geometry type (x/y for grids, element index for dfsu)
- Mesh topology metadata should be included as attributes or coordinate variables (ideally UGRID-compliant)
- CF standard names should be set when the EUM-to-CF mapping is available
- For dfsu, the unstructured dimension does not map to xarray's regular grid model — consider compatibility with xugrid's `UgridDataset` for topology-aware operations
