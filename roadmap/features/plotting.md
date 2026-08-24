---
title: "Plotting"
status: "Delivered"
category: "Visualization"
summary: "Geometry-aware matplotlib plotting for time series, grids, and flexible meshes."
---

## Value Proposition

Visualising simulation results is a core part of any modelling workflow. MIKE IO's plotting dispatches automatically based on geometry type, so users get appropriate visualisations — time series plots for dfs0, pcolormesh for dfs2, triangulated plots for dfsu — without manual setup.

## What This Enables

- **DataArray.plot()**: Single entry point that dispatches to the right plot type based on geometry
- **Grid plots**: 2D filled contour and pcolormesh plots for dfs2 data
- **Mesh plots**: Triangulated plots for dfsu data with element boundaries
- **Time series**: Line plots for dfs0 and point-extracted data
- **Mesh geometry**: Plot mesh structure, bathymetry, and boundary codes
- **Customisable**: All plots return matplotlib axes for further customisation

## Current Status

Delivered. Plotting is integrated into DataArray and geometry classes with automatic dispatch by geometry type.
