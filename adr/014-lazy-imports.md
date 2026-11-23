# ADR-014: Importing MIKE IO Does Not Import matplotlib or scipy

**Status:** Accepted
**Date:** 2026-11

## Context

`import mikeio` takes about 0.85 s. matplotlib accounts for about 0.3 s of that and `scipy.spatial` for about 0.25 s. matplotlib is used only for plotting, scipy only for interpolation and nearest-element search. Reading a file needs neither.

In 2020 this cost did not matter, for three reasons.

- **The import started the .NET runtime.** `mikeio/__init__.py` began with `import clr`. That start-up was the largest part of the import, and no arrangement of Python imports could avoid it.
- **MIKE IO was used in sessions.** A notebook or a long script imports once and then works for minutes. Half a second at the top is not noticed.
- **matplotlib was not imported at module level anyway.** From 0.6 to 1.x the plotting methods imported it inside the function body. That was how the code happened to be written, not a rule, and nothing checked it.

All three have changed.

- mikecore replaced pythonnet ([ADR-002](002-mikecore-over-pythonnet.md)). What remains of the import time is Python packages, and more than half of it is two packages that most calls never use.
- The matplotlib imports moved to module level with the type annotations in 2.0 (2024), because return types name `Axes` and `Figure`. `scipy.spatial` had been at module level since 0.6. Nothing measured import time, so nobody saw the change.
- MIKE IO is now often run by an agent in a chat session. The agent answers a question about a file by writing a short self-contained snippet and running it in a new interpreter, then does the same for the next question. A session is tens of such runs, each paying the full import, with a person waiting for the answer. A cost paid once per session has become a cost paid per question, and for `python -c "import mikeio; mikeio.read('eq.dfs2')"` the two unused packages are more than half of the run time. Scripts started with `uv run` and command-line tools built on MIKE IO pay it the same way.

The language has moved the same way: [PEP 810](https://peps.python.org/pep-0810/) adds lazy imports to Python 3.15 for this reason.

## Decision

`import mikeio`, `mikeio.open()` and `mikeio.read()` do not import matplotlib or scipy. Each is imported by the first call that needs it.

numpy, pandas and mikecore stay at module level. Every read uses them. That includes what mikecore does when imported (about 130 ms, most of it loading the EUM table): with the EUM table not loaded a read returns every unit as undefined, and without the projection library no geometry can be created.

The same applies to any package added later that only some operations need.

**How.**

- A module that needs matplotlib or scipy for some of its functions imports it inside those functions.
- Names needed only for annotations are imported under `TYPE_CHECKING`.
- `DataArray.plot` and `Dataset.plot` are properties that create the plotter on first access.
- The plotter classes exported from `mikeio.dataset` resolve through a module `__getattr__`, so `from mikeio.dataset import DatasetPlotter` still works.

**Check.** A test starts a new interpreter, imports mikeio, reads a dfs0, a dfs2 and a dfsu file, and fails if `matplotlib` or `scipy` is in `sys.modules`.

## Alternatives Considered

**PEP 810 lazy imports** — `lazy import matplotlib` is a syntax error before Python 3.15, and MIKE IO supports 3.12. The compatible form, `__lazy_modules__ = ["matplotlib"]` with an ordinary import statement, is accepted by older versions but ignored: on 3.12 to 3.14 the import stays eager, so nearly all users today would gain nothing. Once 3.15 is the oldest supported version, the function-local imports can be replaced by `lazy import` at module level.

**[lazy_loader](https://scientific-python.org/specs/spec-0001/) (SPEC 1)** — built for a package that delays its own subpackages, with the names declared in a stub file. The problem here is two third-party packages used in a handful of functions. A new dependency and stub files are more than that needs.

## Consequences

- `import mikeio` takes about 0.4 s. The first plot or interpolation in a process pays for its own import.
- matplotlib and scipy remain required dependencies ([ADR-012](012-plotting-scope.md)). This changes when they are loaded, not whether they are installed.
- A broken matplotlib or scipy installation is reported at the first plot or interpolation, not at `import mikeio`.
- `plot` is read-only on `DataArray` and `Dataset`.
- The import contracts of [ADR-013](013-subpackage-boundaries.md) are unaffected: import-linter counts an import inside a function like any other.
- A module-level `import matplotlib` in new code fails the test, not a lint rule, so the failure message has to say what to do.
