# Trade-off Analysis: Public z-coordinate accessor scoping

Refs [#976](https://github.com/DHI/mikeio/issues/976) · companion to [`requirements-clarification.md`](./requirements-clarification.md).

## Executive Summary

Expose the time-varying node and element-center z-coordinates through an **accessor-namespace** on `DataArray` (and `Dataset`), dispatched by `type(self.geometry)` — the exact pattern already used for `DataArray.plot` ([`src/mikeio/dataset/_dataarray.py:264`](../../src/mikeio/dataset/_dataarray.py#L264)). The namespace is always present; on non-layered geometries it raises a clear `AttributeError` with a typed, discoverable message. This keeps `DataArray` a single class (consistent with [ADR-003](../../adr/003-dataset-dataarray-pattern.md) and the [`feedback_no_generic_dataarray`](/home/jan/.claude/projects/-home-jan-src-mikeio/memory/feedback_no_generic_dataarray.md) memory), reuses an established in-tree convention, and adds no churn to the six `return DataArray(...)` sites in slice / arithmetic paths that would have to change if a subclass were introduced. Return type is a raw `np.ndarray` aligned to `da.time`; the namespace exposes `.nodes` and `.elements` as separate properties.

## How to Read This Report

- **Options A1–A4** = scoping mechanisms (where the API lives in the type system).
- **Criteria weights** are 1–3, chosen to reflect what matters for mikeio specifically (see [Criteria](#weighted-criteria)).
- **Scores** are 1 (poor) – 5 (excellent) per criterion. Weighted total is the sum of `weight × score`.
- Evidence is cited inline as `path:line` against the current `main` (commit `4ac8f230`).

## Decision Context

**Core question.** How to publicly expose `DataArray._zn` (`(n_time, n_nodes)` node z-coordinates) when the attribute is meaningful only for layered-derived DataArrays, but `DataArray` is a single class shared across every geometry ([`src/mikeio/dataset/_dataarray.py:91`](../../src/mikeio/dataset/_dataarray.py#L91), `geometry: Any` at L165).

**Constraints.**
- Single `DataArray` class is a long-standing convention ([ADR-003](../../adr/003-dataset-dataarray-pattern.md)); subclassing it has never been done.
- Geometry is composed, not inherited ([ADR-004](../../adr/004-geometry-abstraction.md)); geometry instances are static and time-less.
- File-reader-level subclassing *is* established ([ADR-005](../../adr/005-factory-pattern-open.md): Dfsu2DH vs DfsuSpectral) — but `mikeio.open()` returns reader objects, not DataArrays. The factory does not currently shape DataArray types.
- Six call-sites construct new DataArrays directly with `return DataArray(...)` rather than `return type(self)(...)` ([L494, L728, L1128, L1584, L1699, L1819](../../src/mikeio/dataset/_dataarray.py)); any subclassing approach must address subtype preservation across slicing and arithmetic.
- `_zn` is also populated on 2D vertical-profile DataArrays produced by slicing ([L707–L709](../../src/mikeio/dataset/_dataarray.py#L707)). The accessor must work in both whole-file 3D and slice cases.

**Success criteria.** See the [clarification](./requirements-clarification.md#success-criteria). Most-load-bearing here:
- Non-layered DataArrays fail predictably and discoverably (no silent `None`).
- No data-dependent typing — presence/absence flows from the DataArray's *type or origin*, not from a runtime data check ([`feedback_no_data_dependent_api_behaviour`](/home/jan/.claude/projects/-home-jan-src-mikeio/memory/feedback_no_data_dependent_api_behaviour.md)).
- API consistency with pandas / xarray idioms ([`feedback_api_consistency`](/home/jan/.claude/projects/-home-jan-src-mikeio/memory/feedback_api_consistency.md)).

**Dealbreakers.**
- Returning `None` on non-layered DataArrays (silent, type-eroding).
- Introducing a time-varying `Geometry` abstraction (out of scope per clarification).
- A setter / write API (out of scope per clarification).

## Candidate Options

### A1 — Free function `mikeio.z_coordinates(da)`
A top-level helper that takes a DataArray (or Dataset) and returns the z-array. Raises `TypeError` if the input geometry is not layered. Precedent: [`generic.py`](../../src/mikeio/generic.py) hosts free functions (`concat`, `extract`, `scale`) operating on dfs files.

### A2 — Subclass `DataArrayLayered`
Introduce a subclass returned by Dataset/DataArray construction whenever the underlying geometry is layered. Attribute exists iff the type does — clean for `mypy` and IDE autocomplete. Requires updating every `return DataArray(...)` site to preserve the subtype (6 call sites; `Dataset.__getitem__`; arithmetic; `isel`/`sel`).

### A3 — Accessor namespace `da.z`
A `ZAccessor` object exposed as `da.z`, with `.nodes` and `.elements` properties. The namespace itself is always present, dispatched by `type(self.geometry)` exactly as `da.plot` already is at [`_dataarray.py:264`](../../src/mikeio/dataset/_dataarray.py#L264). On non-layered geometries, `da.z` is a stub that raises `AttributeError("'<Geometry>' has no z-coordinates")` on any access. Mirrors pandas `.dt` / `.str` and xarray's `.cf` accessor.

### A4 — Method on geometry
Add `z_at_nodes(time_index)` / `z_at_elements(time_index)` to `GeometryFM3D`. Caller passes the DataArray's time index. Conceptually misaligned: geometry is time-less by [ADR-004](../../adr/004-geometry-abstraction.md); the data lives on the DataArray. Either geometry must hold a back-reference to its DataArray (breaks composition), or the method takes the z-array as an argument (degenerate — caller already had `_zn`).

## Weighted Criteria

| Criterion | Weight | Why this weight |
|---|---|---|
| Discoverability & type-narrowing | 3 | The whole motivation for moving off `_zn` is making this *findable*. |
| Refactor cost in this PR | 3 | The longer the diff, the more risk of regression in slicing/arithmetic; mikeio v3 is the current stable line. |
| Consistency with mikeio idioms | 3 | Single-class DataArray ([ADR-003](../../adr/003-dataset-dataarray-pattern.md)), composition over inheritance ([ADR-004](../../adr/004-geometry-abstraction.md)), `da.plot` precedent. |
| Pandas / xarray fluency | 2 | Stated user-facing convention ([`feedback_api_consistency`](/home/jan/.claude/projects/-home-jan-src-mikeio/memory/feedback_api_consistency.md)). |
| Future extensibility (dz, layer interfaces, future setter) | 2 | Likely follow-ups; cheap to factor for now. |

Max possible score = `(3+3+3+2+2) × 5 = 65`.

## Comparison Matrix

| Criterion (weight) | A1 free fn | A2 subclass | A3 namespace | A4 geometry method |
|---|---|---|---|---|
| Discoverability (×3) | 2 | 5 | 4 | 3 |
| Refactor cost (×3) | 5 | 1 | 4 | 2 |
| Mikeio idiom fit (×3) | 3 | 2 | 5 | 1 |
| Pandas/xarray fluency (×2) | 2 | 3 | 5 | 1 |
| Extensibility (×2) | 3 | 5 | 5 | 2 |
| **Weighted total / 65** | **39** | **41** | **57** | **23** |

Score justifications:
- **A1 discoverability 2/5**: free functions aren't surfaced on the object; users must know they exist or read docs. Reasonable but worst of the four.
- **A1 refactor 5/5**: zero changes to existing classes.
- **A2 refactor 1/5**: 6 in-class `return DataArray(...)` sites + `Dataset.__getitem__` + arithmetic dunder methods + slicing must all preserve subtype. Either switch to `return type(self)(...)` everywhere (risky for instance creation when type isn't the layered subclass) or branch explicitly. This is a project-wide refactor for one feature.
- **A2 mikeio fit 2/5**: contradicts the single-class DataArray convention. ADR-005's factory-returns-subtype precedent is at the *reader* level, not DataArray.
- **A3 idiom fit 5/5**: directly mirrors `da.plot` at [L264](../../src/mikeio/dataset/_dataarray.py#L264) — same dispatch on `type(self.geometry)`, same shape (namespace object whose capability depends on geometry).
- **A3 discoverability 4/5**: not 5/5 because `da.z` *is* present on every DataArray (mypy will not flag access on non-layered). Mitigation: the stub's `__getattr__` raises with a typed, helpful message; docstring on `da.z` enumerates which geometries support it. Practically as discoverable as `da.plot` is today.
- **A4 idiom fit 1/5**: requires either making geometry time-aware (breaks [ADR-004](../../adr/004-geometry-abstraction.md)) or passing the time array on each call (no improvement over `_zn`).

## Risks & Mitigations

| Option | Top risk | Mitigation |
|---|---|---|
| A1 | Drift: in-tree consumers (plot, writer, export notebook) might keep using `_zn` and the free function becomes a vestigial public veneer. | Migrate all three call-sites in the same PR; remove `_zn` from public sight (single-underscore stays). |
| A2 | Subtype preservation bugs in slicing / arithmetic produce a `DataArray` where a `DataArrayLayered` was expected, silently degrading typing downstream. | Property-based test that every `isel`/`sel`/arithmetic chain from a layered DataArray preserves the subtype. Significant test surface. |
| A3 | `da.z` is present everywhere — users may infer it always works. | (a) Stub raises a clear `AttributeError` with the geometry type in the message; (b) docstring on `ZAccessor` lists supported geometries; (c) user-guide page shows the layered example and the failure mode. Same risk profile as `da.plot`. |
| A4 | Forces geometry to know about time or makes the method useless. | None — the option is dominated by A3 on every criterion. |

## Recommendation

**Adopt A3 — accessor namespace `da.z` with `.nodes` and `.elements` properties.** Weighted total 57/65, dominating A2 (41) and A1 (39).

**Why A3, in one paragraph.** The `da.plot` precedent is the deciding evidence: mikeio already has exactly this pattern — a namespace attribute that dispatches behaviour on `type(self.geometry)` and only "works" for the geometries it supports. Re-using that pattern costs nothing in new conventions and gives users a parallel, learnable surface (`da.plot` for visualization, `da.z` for vertical coordinates). It keeps `DataArray` a single class (ADR-003), keeps geometry static (ADR-004), and avoids the 6-site `return DataArray(...)` refactor that A2 would force. It is also the only option that gives both a typed-feel discoverability (autocomplete shows `da.z.nodes` / `da.z.elements`) and a clear, message-typed failure on non-layered DataArrays.

### Sub-decisions that follow

**Return type — raw `np.ndarray`.** `da.z.nodes` returns the `(n_time, n_nodes)` ndarray directly; `da.z.elements` returns `(n_time, n_elements)`. Callers pair it with `da.time` themselves. This matches how the [Dfsu → netCDF export notebook](https://github.com/DHI/mikeio/blob/main/notebooks/Dfsu%20-%20Export%20to%20netcdf.ipynb) already consumes `_zn`. Wrapping in an `xarray.DataArray` adds a dependency surface and a second time-axis source of truth without earning its complexity for the listed use cases. Revisit if a future caller wants name-based axis indexing.

**Element-center derivation.** Mean of the column-node z's per element, computed lazily on first `.elements` access and cached on the accessor instance. The element table is already available via geometry.

**Naming.** `da.z.nodes` and `da.z.elements`. Sortable together in autocomplete, short, matches the `Node` / `Element` terms in `CONTEXT.md`. José's `z_coordinates` stem is preserved at the namespace level via a longer alias if needed (`da.z_coordinates` → `da.z`), though the recommendation is to ship only `da.z`.

**`Dataset.z`.** Same namespace, returning the shared array (it is identical across items in a Dataset; see [`_dataset.py:331`](../../src/mikeio/dataset/_dataset.py#L331)).

## Implementation Path

1. **Add `ZAccessor` class** in `src/mikeio/dataset/_z_accessor.py` (new file): two cached properties (`nodes`, `elements`), constructor takes the parent DataArray.
2. **Add `NullZAccessor` stub** for non-layered geometries — `__getattr__` raises `AttributeError("DataArray with geometry type {type(geom).__name__} has no z-coordinates")`.
3. **Hook into `DataArray.__init__`**: add `self.z = self._get_z_accessor_by_geometry()`, mirroring the `self.plot = self._get_plotter_by_geometry()` line at [`_dataarray.py:171`](../../src/mikeio/dataset/_dataarray.py#L171). Dispatch uses the same `type(self.geometry)` lookup style as [`_dataarray.py:264`](../../src/mikeio/dataset/_dataarray.py#L264).
4. **Mirror on `Dataset`** as `Dataset.z` returning the same accessor backed by `self[0]._zn`.
5. **Migrate in-tree consumers** (`_data_plot.py:638`, `_dfsu.py:125`, the export notebook) to read `da.z.nodes` instead of `da._zn` — `_zn` stays as a single-underscore implementation detail.
6. **User-guide page** under `docs/user-guide/`: layered z-coordinates, with a worked example (dz-weighted column mean) and the failure mode for non-layered DataArrays.
7. **Tests** in `tests/test_dfsu_layered.py` (or sibling): node and element shapes, slice-derived DataArrays, the `AttributeError` path on a dfs2 / dfsu 2DH DataArray.

Estimated diff: ~200 LOC added, ~20 LOC changed in consumers, plus tests and docs. No structural changes to slicing or arithmetic.

## Confidence

**High.** The decisive evidence is in-repo (`da.plot` pattern), not external; risks are bounded; and the recommendation is the only option that satisfies the dealbreakers without contradicting existing ADRs.

## Alternatives Worth Reconsidering If Conditions Change

- **A2 (subclass)** becomes more attractive *if* mikeio undertakes a wider refactor that subclasses DataArray by geometry (e.g. for typed `.plot` returns, layered-specific arithmetic, or removing the `geometry: Any` escape hatch). At that point the accessor namespace becomes a typed attribute on `DataArrayLayered` for free.
- **A1 (free function)** becomes more attractive *if* the project decides accessor namespaces in general are not a pattern it wants to adopt — but doing so would also mean replacing `da.plot`, which is implausible.
- **Promote z to a first-class wrapped object** (e.g. an `xarray.DataArray` return) *if* a future caller materially benefits from name-indexed axes (CF-compliant export, deeper xarray interop).

## Relationship to PR #850

[PR #850 "Experiment with x,y,z as properties"](https://github.com/DHI/mikeio/pull/850) (closed 2025-10-24) tried introducing a `CoordinateView` namespace on geometry:

```python
geom.nodes.x        # replaces geom.node_coordinates[:, 0]
geom.nodes.y
geom.nodes.z
geom.elements.x     # replaces geom.element_coordinates[:, 0]
geom.elements.y
geom.elements.z
```

It was closed because the migration cost (>20 internal call-sites + public-API churn) outweighed the ergonomic gain over the established `[:, 0/1/2]` indexing — **not** because the namespace pattern itself was rejected.

This bears on the present trade-off:

- **Different orthogonalization.** #850's namespace puts location outermost (`geom.nodes.x`), axis innermost. The present spec puts quantity outermost (`da.z.nodes`), location innermost. If #850 (or a successor) is ever revived, the two surfaces will not compose along a single "outer key" convention.
- **Asymmetry accepted deliberately.** The dynamic z accessor is z-only and time-aware, so an outer `da.z` reads more naturally than `da.nodes.z` (which would imply a peer `da.nodes.x` and `da.nodes.y` that have no time variance). The two namespaces measure different things and the structural mismatch is honest.
- **No dependency on #850.** The `da.z` accessor is independently coherent; it neither requires nor blocks a future static-coords namespace.

## Naming Convention Note

The existing `geom.node_coordinates` / `geom.element_coordinates` attributes (full `(n, 3)` xyz arrays, static, on **geometry**) establish a `*_coordinates` suffix convention. The dynamic z-coordinate accessor consciously does **not** extend that convention by name, for these reasons:

- **Different home**: the static analogs belong on geometry (mesh property); the dynamic z's belong on the DataArray (carries the time axis). ADR-004 deliberately keeps geometry time-less, so naming parity would imply location parity that the architecture rules out.
- **Different shape and content**: `node_coordinates` is `(n_nodes, 3)` xyz; the dynamic version is z-only and time-aware (`(n_time, n_nodes)`). The asymmetry mirrors real underlying-data asymmetry; forcing symmetric names would obscure it.
- **Different pattern precedent**: `da.plot` is the in-repo prior for "geometry-dispatched attribute on DataArray", and that is the pattern this feature extends.

The direct-attribute alternative `da.node_z_coordinates` / `da.element_z_coordinates` was considered (weighted score 50/65 vs A3's 57/65) and rejected: it carries name-parallelism with the static pair but gives up the namespace as a future home for related z-axis quantities (column thickness `dz`, layer-interface elevations, etc.), and `mikeio` does not otherwise pursue rigorous attribute-naming symmetry across its data and geometry surfaces.

The static `geom.node_coordinates` / `geom.element_coordinates` attributes stay untouched. They are widely used internally (>20 sites in `_FM_geometry_layered.py` alone), correctly located on geometry, and their full-xyz shape is appropriate for static mesh data.

## Decisions Log

| Decision | Rationale | Date |
|---|---|---|
| Adopt accessor-namespace `da.z` (option A3) | Mirrors existing `da.plot` dispatch by geometry type; keeps `DataArray` a single class; lowest refactor cost; clear failure mode on non-layered. Weighted total 57/65. | 2026-05-19 |
| Return raw `np.ndarray`, not xarray.DataArray | Matches how `_zn` is consumed today; avoids a second time-axis source of truth; no new dependency surface. | 2026-05-19 |
| Expose `nodes` and `elements` from day one | Asked for in [#976](https://github.com/DHI/mikeio/issues/976) by José Antonio Arenas; element-center is a lazy-cached derivation, marginal cost. | 2026-05-19 |
| Defer setter and time-varying-geometry refactor | Out of scope per [clarification](./requirements-clarification.md); revisit only if conditions in "Alternatives" above change. | 2026-05-19 |
| Keep static `geom.node_coordinates` / `geom.element_coordinates` untouched; do not rename or extend them | Widely used internally; correctly located on geometry (ADR-004 keeps geometry time-less); full-xyz shape is right for static mesh data. The dynamic z-accessor's name asymmetry is acceptable and reflects real ontological asymmetry. | 2026-05-19 |
| Adopt the `da.z` namespace despite breaking the `*_coordinates` suffix convention | Convention parallelism considered (Option α: `da.node_z_coordinates` / `da.element_z_coordinates`, weighted 50/65). The namespace's extensibility for related z-axis quantities (`dz`, layer-interface elevations) and its parallelism with `da.plot` outweigh the lost name-parallelism with the static analog. | 2026-05-19 |
