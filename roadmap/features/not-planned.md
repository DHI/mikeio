---
title: "Not Planned"
status: "Not Planned"
category: "Out of Scope"
summary: "Features that have been considered and determined to be outside MIKE IO's scope."
---

## Horizontal Slicing of 3D dfsu at Specified Depth

Extracting a horizontal slice at an arbitrary depth from a 3D layered mesh requires interpolation between layers — a non-trivial geometric operation that depends on the vertical discretisation scheme. This is available in MIKE's proprietary tooling which handles the interpolation correctly for all mesh configurations.

## Vertical Slicing of 3D dfsu Along Waypoints

Creating a vertical cross-section along a polyline path through a 3D mesh requires intersecting the path with element boundaries, interpolating values, and constructing a new 2D representation. This is a complex geometric operation available in MIKE's proprietary tools.

## Vertical Aggregation of 3D dfsu Over a Depth Range

Aggregating values over a specific depth range (e.g., averaging salinity from 0–10m) requires identifying which layers intersect the range, computing partial volumes for boundary layers, and performing volume-weighted aggregation. This is available in MIKE's proprietary tooling.

## Creation of 3D dfsu Transect from External Data

Constructing a 3D transect file suitable for use as a boundary condition requires building a valid mesh section with correct layer structure and connectivity. This is specialised model setup functionality available in MIKE's proprietary tools.

## Comparing Scenarios with Different Meshes

Interpolating results from one unstructured mesh onto another — necessary for comparing model runs with different mesh resolutions — requires robust spatial interpolation in 2D or 3D. This is a complex geometric problem handled by MIKE's proprietary interpolation engine.

## Discharge Calculations Across a Transect by Layer

Computing discharge (flux) across a cross-section requires face-based velocity data, correct normal vector computation, and layer-aware integration. This specialised hydrodynamic analysis is available in MIKE's proprietary post-processing tools.

## Brunt-Väisälä Frequency

Computing the buoyancy frequency (N²) from temperature and salinity profiles is a domain-specific oceanographic analysis rather than a file I/O operation. This would be a great fit for a dedicated ocean/coastal science library that builds on MIKE IO's data structures. See the [feature page](brunt-vaisala.md) for details.

## Ensemble Model Support

DFS file formats store data on fixed spatial and temporal axes — they have no concept of an ensemble dimension. Supporting ensembles would require either a fundamentally different file format or a higher-level abstraction that manages collections of files. This is a format limitation rather than a MIKE IO limitation.
