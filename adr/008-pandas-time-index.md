# ADR-008: Use pandas for Time Handling

**Status:** Accepted
**Date:** 2020

## Context

Time series data requires robust timestamp handling and flexible subsetting. Building this from scratch is error-prone.

## Decision

Use pandas `DatetimeIndex` (naive, no time zone) for all time handling in Dataset and DataArray.

## Alternatives Considered

- **Custom implementation**: Would require reimplementing and testing time logic for edge cases.
- **numpy datetime64**: Limited support for flexible string-based slicing.

## Consequences

- Simple, expressive time subsetting: `ds.sel(time="2020-01-15")`, `ds.sel(time=slice("2020-01", "2020-06"))`
- Leverage pandas' mature, well-tested time handling
- pandas is already a common dependency in scientific Python
