# ADR-004: Geometry Abstraction

**Status:** Accepted
**Date:** 2022 (evolved through v2.0)

## Context

Geometry logic was embedded in file reader classes (Dfs2, Dfsu, etc.), making the readers complex and geometry operations hard to reuse or test independently.

## Decision

Extract geometry into standalone classes:
- **Grid1D, Grid2D, Grid3D**: Structured grids
- **GeometryFM2D, GeometryFM3D**: Flexible mesh (unstructured) geometries

These classes are composed into Dataset/DataArray rather than inherited.

## Alternatives Considered

- **Keep geometry in file readers**: Simpler initially but leads to code duplication and large classes.
- **Inheritance hierarchy**: More rigid; composition provides better flexibility.

## Consequences

- Geometry can be shared across multiple datasets
- Easier to test geometry operations in isolation
- File readers become simpler (focused on I/O)