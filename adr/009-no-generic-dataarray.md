# ADR-009: No Generic DataArray/Dataset Over Geometry Type

**Status:** Accepted
**Date:** 2025-03

## Context

With 14 geometry types unified under the `GeometryType` union, it's natural to consider making `DataArray` and `Dataset` generic: `DataArray[T]` / `Dataset[T]` where `T` is bound to `_Geometry`. This would let type information flow from file readers (e.g., `Dfs2.read() -> Dataset[Grid2D]`) through downstream operations without isinstance narrowing.

All geometry types inherit from the `_Geometry` ABC, so `TypeVar("T", bound=_Geometry)` is straightforward. Geometry-preserving operations (arithmetic, copy, time-axis aggregation) could return `DataArray[T]`, preserving the known type.

## Decision

Don't make DataArray or Dataset generic over geometry type. Keep using the `GeometryType` union with isinstance narrowing.

## Why

**The type parameter loses its value quickly.** The most common operations after reading — `isel`, `sel`, spatial aggregation — change the geometry type at runtime:

- `Grid2D.isel(x=5)` → `Grid1D`
- `GeometryFM3D.isel(layer=0)` → `GeometryFM2D`
- `GeometryFM2D.isel(element=42)` → `GeometryPoint2D`
- `da.mean(axis="space")` → `Geometry0D`

So `T` is only useful in the narrow window between reading and the first spatial operation. After that, the type widens back to the full union.

**Fixing this would mean letting the type system drive API design.** To keep `T` narrow, we'd either need:

1. Complex `@overload` chains for every geometry transition (e.g., `isel(self: DataArray[Grid2D], x: int) -> DataArray[Grid1D]`), adding Rust-like type complexity.
2. Changing `isel`/`sel` to preserve geometry type (a Grid2D with nx=1 stays Grid2D instead of becoming Grid1D), which breaks numpy/xarray convention where scalar indexing drops a dimension.

Both options reshape the API or its complexity to serve the type system rather than the user.

**Permanent complexity tax.** `Generic[T]` touches ~50 method signatures in DataArray, ~30 in Dataset, plus all file readers. Every future function that accepts or returns DataArray must decide how to handle the type parameter.

## Alternatives Considered

- **Full `Generic[T]` with geometry-changing methods returning `DataArray[GeometryType]`**: Technically works but provides minimal benefit — the type parameter widens at the first `isel`/`sel` call.
- **`@overload` for geometry transitions**: Correct types for `isel` on each geometry, but requires many overloads and is fragile to maintain.
- **Preserve geometry type in `isel`/`sel`**: Would make generics trivial but breaks user expectations from numpy/xarray.

## Consequences

- `DataArray.geometry` remains typed as `GeometryType` (the 14-member union)
- Geometry-specific code uses isinstance narrowing, which mypy handles correctly
- The API stays simple and Pythonic — no generic type parameters to reason about
- File reader return types remain `-> Dataset` without type parameters
