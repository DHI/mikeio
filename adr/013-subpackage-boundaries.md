# ADR-013: Subpackages Are Imported Through Their `__init__`

**Status:** Accepted
**Date:** 2026-11

Extends [ADR-011](011-import-layering.md).

## Context

ADR-011 fixed the order between the top-level modules: `dfsu` may import `dfs`, never the reverse. It says nothing about *what* in `dfs` may be imported. `dfsu` and `generic` take seven helpers straight from `dfs/_dfs.py`; `dataset` and `dfsu` import from four files inside `spatial`. All of it is downward, so the layers contract passes.

Each such import ties a file outside the subpackage to a file inside it. Renaming, splitting or rewriting `_dfs.py` or `_FM_geometry.py` then starts with a search of the whole package for users, and nothing reports when a new user is added. `dataset`, `dfsu`, `eum` and `pfs` had no outside importers of their inner files, but only by habit.

Two more properties held by habit. `dataset` and the modules below it never import mikecore, which is what keeps mikecore types out of `Dataset` and `DataArray`. `pfs` imports nothing from the data modules and can be read and changed on its own.

## Decision

Add three kinds of contract to `.importlinter`, checked by `just layers` with the layers contract.

**Subpackage boundaries.** One `protected` contract per subpackage (`dataset`, `dfs`, `dfsu`, `eum`, `pfs`, `spatial`): code outside the subpackage imports from `mikeio.<subpackage>`, never from a file inside it. Files inside a subpackage import each other freely. The 16 imports that cross a boundary today are listed as exceptions in the `dfs` and `spatial` contracts, each group with the reason it exists.

**Third-party packages.** mikecore is not imported by `dataset`, `_track`, `_options` or the foundation modules. matplotlib is not imported by the file modules (`dfs`, `dfsu`, `eum`, `generic`). Only direct imports are checked: `dataset` still reaches mikecore through `spatial` and `eum`.

**`pfs` stands alone.** `pfs` imports no data module, mikecore or matplotlib.

The layers contract becomes exhaustive, so a new top-level module fails the check until it is given a layer.

Each contract carries the fix in its failure message: import from the subpackage, or export the name there, and do not extend the exception list.

## Alternatives Considered

**Move the code first, then add contracts with no exceptions** — the shared dfs helpers would go to a module below `dfs`, the flexible mesh plotting functions out of `spatial`, and `Area` and the mesh base classes would be exported. That removes the exceptions but moves private modules that tests and downstream code import, shortly before a release that is meant to remove things, not rearrange them. Listing the exceptions stops new ones now; the moves can follow when the geometry base class is made public.

**Apply the same rule to tests** — ruff's `import-private-name` rule flags any import through an underscore module. A unit test of an internal module such as `_interpolation` or `_time` is worth having and has to import it, so the rule would remove useful tests or need an exception per file.

## Consequences

- An import of a file inside another subpackage fails CI. The fix is to export the name from the subpackage or to move the code, which is a decision about where the code belongs rather than a formality.
- The exception lists in the `dfs` and `spatial` contracts are the known cost of changing `_dfs.py`, `_FM_geometry.py` and `_FM_plot.py`. They should shrink, not grow.
- Tests are not covered. They may import internal modules; public classes should still come from their public path.
- `spatial` imports mikecore in six modules and is left out of the mikecore contract.
- `include_external_packages` is switched on, which the contracts naming mikecore and matplotlib need.
