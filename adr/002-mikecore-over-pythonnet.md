# ADR-002: Migrate from pythonnet to mikecore

**Status:** Accepted
**Date:** 2021-12

## Context

MIKE IO originally used pythonnet to call .NET DFS libraries. This caused cross-platform issues (Linux support) and delays supporting new Python versions.

## Decision

Create mikecore-python: rewrite the core DFS I/O functionality from .NET to C/C++ with pybind11 bindings.

## Alternatives Considered

- **Keep pythonnet**: Would maintain status quo issues with cross-platform and Python version support.
- **Pure Python implementation**: Would require reimplementing DFS format parsing; significant effort and risk of incompatibilities.

## Consequences

- Cross-platform support (Windows, Linux)
- Faster adoption of new Python versions
- Simpler installation via pip
- Significant upfront investment to rewrite core library in C/C++
