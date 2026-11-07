---
title: "Spatial Resampling of dfs2"
status: "Under Consideration"
category: "Grid Operations"
summary: "Resample dfs2 grids to different resolutions or extents using interpolation."
---

## Value Proposition

Modellers frequently need to change the spatial resolution of gridded data — coarsening high-resolution bathymetry for a larger-domain model, or refining a coarse forcing field for a detailed local model. Spatial resampling of dfs2 files would support these workflows natively.

## What This Enables

- **Upsampling**: Increase grid resolution using interpolation (bilinear, nearest-neighbour)
- **Downsampling**: Reduce grid resolution using aggregation (mean, max, min)
- **Regridding**: Resample to a different grid extent or origin
- **Resolution matching**: Align grids from different sources to a common resolution

## Current State

No spatial resampling support exists. Grid2D geometry is defined in `_grid_geometry.py` and dfs2 I/O in `_dfs2.py`, but there are no interpolation or aggregation operations on grids.
