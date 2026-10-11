# ADR-011: Enforced Import Layering

**Status:** Accepted
**Date:** 2026-09

## Context

MIKE IO's modules have an order to them — `eum` knows nothing about `dfsu`, and `dfsu` builds on `dfs` and `dataset` — but nothing recorded or checked it. A module could import any other, and the only way to find out whether the structure still held was to read the imports.

It had already started to give. Eight modules imported `__dfs_version__`, `Dataset`, or `__version__` from the root package (`from .. import X`) instead of from the module that defines it. Since `mikeio/__init__.py` imports the whole package, each of those is an edge to everything: `dfsu` reached `pfs`, `dfs` reached `dfsu`, all through `__init__`, hiding what the real dependency was.

Two dependencies point genuinely upward and are load-bearing rather than accidental: `Dataset.to_dfs()` dispatches on the output suffix, reaching into the writer of every format including `dfsu`; track extraction takes either a dfs0 file or a `Dataset` and returns a `Dataset`. Both are deferred into the function body, which keeps the module importable but hides the dependency from the top of the file.

## Decision

Write the layering down in `.importlinter` and check it with [import-linter](https://import-linter.readthedocs.io/) on every build (`just layers`, part of `just check`).

Modules may import downward and never upward. Modules sharing a layer may import each other.

```mermaid
flowchart TD
    subgraph L1["Top-level API"]
        generic
        pfs
    end
    subgraph L2["Flexible mesh files"]
        dfsu
    end
    subgraph L3["Grid and time series files"]
        dfs
    end
    subgraph L4["Data structures"]
        dataset
    end
    subgraph L5["Track extraction"]
        _track
    end
    subgraph L6["Geometry"]
        spatial
    end
    subgraph L7["Runtime options"]
        _options
    end
    subgraph L8["Units"]
        eum
    end
    subgraph L9["Foundations"]
        exceptions
        _path
        _interpolation
        _spectral
        _time
    end
    generic --> dfs
    generic --> eum
    pfs --> _path
    dfsu --> dfs
    dfsu --> dataset
    dfsu --> _track
    dfs --> dataset
    dataset --> _track
    dataset -. "to_dfs() dispatch" .-> dfs
    dataset -. "to_dfs() dispatch" .-> dfsu
    _track --> spatial
    _track -. "dfs0 or Dataset input" .-> dataset
    _track -. "dfs0 or Dataset input" .-> dfs
    dfs --> spatial
    dfsu --> spatial
    spatial --> eum
    _options --> eum
    eum --> exceptions
```

Each box is a layer, named for what it contributes; modules inside a layer may import each other. Arrows point from importer to imported, and transitively implied edges are omitted. The two dotted pairs are the accepted violations.

## Alternatives Considered

**Leave it to review** — the eight root-package imports all passed review; a reviewer sees one import, not what it does to the graph.

**Independence contracts instead of layers** — these forbid modules importing each other without saying which direction is correct. Layers state the intended shape, so a new module has somewhere to belong.

**Fix `to_dfs()` and track extraction to remove the upward imports** — needs a registry of writers/readers keyed by suffix, decoupling `dataset` from the concrete file-format modules. A larger change than this ADR is about; recording the two as accepted exceptions makes them visible without forcing that redesign.

## Consequences

- A new module has to be placed in a layer, which is the question worth asking when adding one.
- Type-only imports are excluded (`exclude_type_checking_imports`), so an annotation may point upward.
- The ignore list is the debt list. It is seven lines, expanding two conceptual exceptions across the files each touches; if it grows beyond that it's a sign the layering, or the code, needs to change.
- `from .. import X` / `from mikeio import X` across a layer boundary is now a violation — it names the re-export rather than the module that defines it.
- import-linter is a test-group dependency, next to mypy, since that is what CI installs.
