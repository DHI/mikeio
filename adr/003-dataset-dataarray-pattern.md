# ADR-002: Dataset and DataArray Pattern

**Status:** Accepted
**Date:** 2020-10

## Context

Early versions returned raw numpy arrays. Users needed to keep data together with metadata (time, items) and crucially, the spatial geometry.

## Decision

Introduce Dataset and DataArray classes inspired by xarray:
- **DataArray**: Single variable with data, time, geometry, and item metadata
- **Dataset**: Collection of DataArrays sharing the same time and geometry

The geometry is a first-class citizen, enabling spatial operations (selection, interpolation) directly on data objects.

## Alternatives Considered

- **Return xarray directly**: xarray lacks native support for unstructured meshes and MIKE geometries.
- **Return raw numpy arrays**: No metadata; geometry must be tracked separately.

## Consequences

- Spatial operations (e.g., `sel(x=..., y=...)`) work directly on data
- Clean separation between file I/O and data manipulation
- Easy conversion to xarray/pandas when needed
