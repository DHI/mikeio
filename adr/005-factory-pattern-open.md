# ADR-004: Factory Pattern for File Opening

**Status:** Accepted
**Date:** 2020

## Context

MIKE files come in many formats (dfs0, dfs1, dfs2, dfs3, dfsu with various subtypes). Users shouldn't need to know which specific class to instantiate.

## Decision

Provide `mikeio.open()` as a single entry point that inspects the file and returns the appropriate handler (Dfs0, Dfs2, Dfsu2DH, DfsuSpectral, etc.).

## Alternatives Considered

- **Require users to pick the class**: Error-prone; dfsu alone has multiple subtypes that are hard to distinguish.
- **Single class for all formats**: Would lead to a monolithic class with many conditional branches.

## Consequences

- Simple, discoverable API: `mikeio.open("file.dfsu")`
- File type detection is centralized
- New file types can be added without changing user code
