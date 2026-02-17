# ADR-007: PFS Module

**Status:** Accepted
**Date:** 2020

## Context

MIKE models use PFS (Parameter File System) files for configuration. Working with MIKE models requires reading and modifying these files alongside DFS result files.

## Decision

Include PFS support in MIKE IO with attribute-style access to sections and parameters:

```python
pfs = mikeio.read_pfs("model.m21fm")
value = pfs.FemEngineHD.TIME.start_time
```

## Alternatives Considered

- **DHI PFS SDK**: The official [PFS API](https://docs.mikepoweredbydhi.com/core_libraries/pfs/pfs_api/) requires .NET/C# knowledge and uses OOP patterns (PFSFile, PFSBuilder classes) that are less accessible to modelling engineers without software development background.

## Consequences

- Pythonic API accessible to modelling engineers
- Intuitive navigation via attribute access mirrors the PFS file structure
- Enables automation of MIKE model configuration
- Cross-platform (no .NET dependency)
