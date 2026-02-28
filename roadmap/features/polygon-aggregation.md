---
title: "Horizontal Aggregation in Polygons"
status: "Under Consideration"
category: "Analysis"
summary: "Compute area-weighted zonal statistics (mean, sum, min, max) for elements within polygons."
---

## Value Proposition

Engineers often need spatially aggregated values within specific zones — average salinity in a harbour basin, total discharge through an inlet, or maximum wave height in a protected area. Polygon-based aggregation provides these results directly without manual element selection and weighting.

## What This Enables

- **Zonal statistics**: Compute area-weighted mean, sum, min, max within arbitrary polygons
- **Multi-zone analysis**: Aggregate across multiple polygons simultaneously (e.g., management zones)
- **Time series extraction**: Produce a single time series per zone from a spatial field

## Current State

Polygon-based element selection exists in `GeometryFM2D` (find elements within a polygon), and element areas are available. The building blocks are in place, but there is no high-level API that combines selection with area-weighted aggregation.
