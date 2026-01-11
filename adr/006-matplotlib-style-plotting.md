# ADR-005: Matplotlib-Style Plotting API

**Status:** Accepted
**Date:** 2021

## Context

Users need to quickly visualize MIKE data. Many are familiar with pandas and xarray where plotting is available via `.plot()` accessor.

## Decision

Provide a `.plot` accessor on DataArray and Dataset that follows the pandas/xarray convention, delegating to matplotlib.

Example: `da.plot()` produces a sensible default plot based on the data's geometry.

## Alternatives Considered

- **Standalone plotting functions**: Less discoverable; breaks the fluent API pattern.
- **Custom plotting library**: Unnecessary complexity; matplotlib is the standard.

## Consequences

- Familiar API for pandas/xarray users
- Quick data exploration without boilerplate
- Plot type adapts to geometry (line plot for 1D, pcolormesh for 2D, etc.)
- Matplotlib remains an optional dependency for core functionality
