# ADR-001: NumPy as Data Backend

**Status:** Accepted
**Date:** 2020

## Context

MIKE IO needs an in-memory representation for numerical data from DFS files.

## Decision

Use numpy arrays as the underlying data container in DataArray.

## Alternatives Considered

- **Custom array class**: No benefit; would limit interoperability.
- **xarray directly**: Adds heavy dependency; less control over MIKE-specific behavior.

## Consequences

- Direct compatibility with scientific Python ecosystem (scipy, scikit-learn, matplotlib)
- Users can apply numpy operations directly to data
- Minimal learning curve for Python users
