# FIS: Public z-coordinate accessor for layered dfsu DataArrays

Refs [#976](https://github.com/DHI/mikeio/issues/976) · companion to [`requirements-clarification.md`](./requirements-clarification.md) and [`tradeoff.md`](./tradeoff.md).

## Feature Overview and Goal

**Intent**: Expose the time-varying node and element-center z-coordinates of layered dfsu DataArrays through a public accessor namespace `da.z`, dispatched by geometry type — replacing direct reliance on the private `_zn` attribute.

**Expected Outcomes**:

- [OC01] Users access node and element-center z-coordinates of layered DataArrays / Datasets through `da.z.nodes` and `da.z.elements`, without touching any `_zn`.
- [OC02] DataArrays whose geometry is not layered fail predictably and discoverably when `da.z` is accessed — clear `AttributeError` naming the geometry type, no silent `None`.
- [OC03] In-tree consumers (vertical-profile plot, 3D writer, the Dfsu → netCDF export notebook) read z-coordinates through the public accessor.
- [OC04] Existing tests that previously asserted on `_zn` are migrated to the public API.


## Required Context

### From [`tradeoff.md`](./tradeoff.md) – "Recommendation"
<!-- source: specs/issue-976-z-coordinates-public-api/tradeoff.md#recommendation -->
<!-- extracted: 2026-05-19 -->

> Adopt A3 — accessor namespace `da.z` with `.nodes` and `.elements` properties. The `da.plot` precedent is the deciding evidence: mikeio already has exactly this pattern — a namespace attribute that dispatches behaviour on `type(self.geometry)` and only "works" for the geometries it supports. Re-using that pattern costs nothing in new conventions and gives users a parallel, learnable surface (`da.plot` for visualization, `da.z` for vertical coordinates). It keeps `DataArray` a single class (ADR-003), keeps geometry static (ADR-004), and avoids the 6-site `return DataArray(...)` refactor that A2 would force. Return type is a raw `np.ndarray`; element-center is the mean of column-node z's per element, computed lazily on first `.elements` access and cached on the accessor instance.

### From [`requirements-clarification.md`](./requirements-clarification.md) – "Scope"
<!-- source: specs/issue-976-z-coordinates-public-api/requirements-clarification.md#scope -->
<!-- extracted: 2026-05-19 -->

> - A public, read-only way to obtain the time-varying node z-coordinates currently held in `DataArray._zn` / `Dataset._zn`.
> - A separately-exposed element-center z-coordinate view derived from the node values.
> - Coverage of every DataArray instance where `_zn` is currently populated — 3D layered dfsu and the 2D vertical-profile / column-slice DataArrays produced from them.
> - **Out of scope**: setter / write path; time-varying geometry abstraction; CF-conformant NetCDF export tooling.


## Deeper Context

- [`tradeoff.md`](./tradeoff.md) – full design-space analysis, including the rejected alternatives (free function, subclass, geometry-method) and why each loses.
- [`requirements-clarification.md`](./requirements-clarification.md) – open design questions (now resolved) and the decisions log.
- [PR #850 "Experiment with x,y,z as properties"](https://github.com/DHI/mikeio/pull/850) – closed experiment with a `geom.nodes.x` / `geom.elements.x` namespace on geometry. Closed for migration cost, not pattern fit. See the *Relationship to PR #850* section of the trade-off for how it bears on the orthogonalization choice here.
- [`adr/003-dataset-dataarray-pattern.md`](../../adr/003-dataset-dataarray-pattern.md) – single-class DataArray convention that A3 preserves.
- [`adr/004-geometry-abstraction.md`](../../adr/004-geometry-abstraction.md) – geometry as composed, static state.
- `src/mikeio/dataset/_dataarray.py#_get_plotter_by_geometry` (L264) – the pattern this feature mirrors.
- [Dfsu → netCDF export notebook](https://github.com/DHI/mikeio/blob/main/notebooks/Dfsu%20-%20Export%20to%20netcdf.ipynb) – downstream consumer to migrate.


## Acceptance Scenarios

- [x] **S01 [OC01] [TI01,TI03] `da.z.nodes` on a 3D layered DataArray returns the per-timestep node z-coordinates**
  - **Given** a DataArray read from a 3D layered dfsu file (e.g. `oresund_sigma_z.dfsu`)
  - **When** the caller accesses `da.z.nodes`
  - **Then** the result is an `np.ndarray` of shape `(n_time, n_nodes)`, equal element-wise to the legacy `da._zn`, with values aligned to `da.time`

- [x] **S02 [OC01] [TI01,TI03] `da.z.elements` returns the per-timestep element-center z-coordinates**
  - **Given** a DataArray read from a 3D layered dfsu file
  - **When** the caller accesses `da.z.elements`
  - **Then** the result is an `np.ndarray` of shape `(n_time, n_elements)` whose values equal the mean of each element's node z-coordinates at each timestep, and subsequent accesses return the same cached array without recomputation

- [x] **S03 [OC01] [TI01,TI03] Vertical-profile slice DataArray retains the accessor**
  - **Given** a 3D layered DataArray reduced to a vertical column via `sel(x=, y=)` (the path that populates `_zn` at [`_dataarray.py:707-709`](../../src/mikeio/dataset/_dataarray.py#L707))
  - **When** the caller accesses `da.z.nodes` on the slice
  - **Then** the array has shape consistent with the slice's node count and matches the slice's legacy `_zn`

- [x] **S04 [OC02] [TI02,TI03] Non-layered DataArray raises an attribute error with the geometry type named**
  - **Given** a DataArray whose geometry is not layered (e.g. dfs2 grid, dfsu 2DH, spectral)
  - **When** the caller accesses `da.z.nodes` (or any attribute on `da.z`)
  - **Then** an `AttributeError` is raised whose message includes the concrete geometry type name and explains that this DataArray has no z-coordinates

- [x] **S05 [OC01] [TI04] `Dataset.z` mirrors per-DataArray accessor backed by the shared array**
  - **Given** a Dataset read from a 3D layered dfsu file
  - **When** the caller accesses `ds.z.nodes` and `ds[0].z.nodes`
  - **Then** both return arrays of identical shape and value, consistent with the existing `_zn` invariant at [`_dataset.py:331`](../../src/mikeio/dataset/_dataset.py#L331)

- [x] **S06 [OC03,OC04] [TI05,TI06,TI08] In-tree consumers and existing layered tests use the public accessor**
  - **Given** the vertical-profile plotter, the 3D dfsu writer, and the existing layered tests at [`tests/test_dfsu3d.py:183-194`](../../tests/test_dfsu3d.py#L183) and [L605-607](../../tests/test_dfsu3d.py#L605)
  - **When** the codebase is grepped for `\._zn\b` outside `src/mikeio/dataset/`
  - **Then** no occurrences remain — every reader has migrated to `da.z.nodes` (or `da.z.elements` where appropriate)


## Structural Criteria

- [x] `ruff check .` and `ruff format --check .` pass.
- [x] `mypy .` passes; the new accessor classes are fully typed (no `Any`).
- [x] `_zn` remains as a private (single-underscore) implementation detail on `DataArray` and `Dataset`; nothing in this PR removes it, and external consumers should not be required to touch it.
- [x] No new public symbol is exported beyond what this FIS specifies (the `ZAccessor` class need not be public — only the `.z` attribute on `DataArray` / `Dataset`).


## Scope & Boundaries

### Work Areas
- **New module** `src/mikeio/dataset/_z_accessor.py` — defines `ZAccessor` (for layered geometries) and `NullZAccessor` (for everything else).
- **`DataArray.__init__` hook** in `src/mikeio/dataset/_dataarray.py` — add `self.z = self._get_z_accessor_by_geometry()` next to the existing `self.plot = self._get_plotter_by_geometry()` line at L171; add the dispatcher method mirroring [`_dataarray.py:264`](../../src/mikeio/dataset/_dataarray.py#L264).
- **`Dataset.z` property** in `src/mikeio/dataset/_dataset.py` — return the accessor from `self[0]`, mirroring the `_zn` delegation at [L331](../../src/mikeio/dataset/_dataset.py#L331).
- **Consumer migration** — `src/mikeio/dataset/_data_plot.py:638`, `src/mikeio/dfsu/_dfsu.py:125`, and `notebooks/Dfsu - Export to netcdf.ipynb` switch from `_zn` to `da.z.nodes`.
- **Test migration** — `tests/test_dfsu3d.py` lines 183-194 and 605-607 replace `_zn` asserts with `.z.nodes` asserts (paying down `feedback_no_private_asserts` debt for these cases).
- **User-guide page** — new `docs/user-guide/dfsu-3d.qmd` section (or extension of the existing 3D page) showing `da.z.nodes` / `da.z.elements` with a worked dz-weighted column-mean example.

### What We're NOT Doing
- **A setter / write path** — out of scope per the clarification; the writer continues to read `_zn` internally in this PR.
- **A time-varying geometry abstraction** — out of scope; geometry stays static (ADR-004).
- **CF-conformant NetCDF export helpers** — out of scope; this PR only makes them easier to build later.
- **Removing `_zn`** — kept as a private alias for one release cycle to avoid churning unrelated internal paths in this PR.
- **Exposing the `ZAccessor` class as a public symbol** — only `da.z` / `ds.z` are part of the public surface; the class is an implementation detail.


## Architecture Decision

**Approach**: Accessor namespace `da.z` dispatched by `type(self.geometry)`, mirroring the existing `da.plot` pattern at [`_dataarray.py:264`](../../src/mikeio/dataset/_dataarray.py#L264). See [`tradeoff.md`](./tradeoff.md).
**Why this over alternatives**: Reuses an established in-tree dispatch convention, keeps `DataArray` a single class (ADR-003), avoids the six-site `return DataArray(...)` refactor that a `DataArrayLayered` subclass would force.


## Code Patterns & External References

```
# type | path#anchor or url                                              | why needed (intent)
file   | src/mikeio/dataset/_dataarray.py#_get_plotter_by_geometry       | Mirror this dispatch pattern for the z-accessor (type(self.geometry) lookup)
file   | src/mikeio/dataset/_dataarray.py#L171                           | Hook site for self.z = self._get_z_accessor_by_geometry()
file   | src/mikeio/dataset/_dataarray.py#L707                           | Vertical-profile slice path that propagates _zn — must keep working
file   | src/mikeio/dataset/_dataset.py#L331                             | Pattern for Dataset-level delegation to self[0]
file   | src/mikeio/dataset/_data_plot.py#L638                           | Vertical-profile plotter — consumer to migrate
file   | src/mikeio/dfsu/_dfsu.py#L125                                   | 3D writer — consumer to migrate
file   | tests/test_dfsu3d.py#L183                                       | Existing _zn asserts to migrate
file   | tests/testdata/oresund_sigma_z.dfsu                             | Layered-with-varying-sigma fixture (sigma-z hybrid)
file   | tests/testdata/basin_3d.dfsu                                    | Layered fixture for basic 3D cases
url    | https://github.com/DHI/mikeio/blob/main/notebooks/Dfsu%20-%20Export%20to%20netcdf.ipynb | External consumer to migrate
```


## Constraints & Gotchas

- **Avoid**: subclassing `DataArray` to encode layered-ness — every `return DataArray(...)` site at L494, L728, L1128, L1584, L1699, L1819 would need to preserve the subtype. **Instead**: dispatch the accessor on geometry type via the existing `da.plot`-style mapping.
- **Critical**: `_zn` shape after slicing is *not* always 2D — at [`_dataarray.py:707-709`](../../src/mikeio/dataset/_dataarray.py#L707) it becomes 1D when `_zn.ndim == 2` is false. The `ZAccessor.nodes` property must surface the slice's `_zn` faithfully without re-imposing `(n_time, n_nodes)` shape assumptions.
- **Constraint**: element-center derivation needs the element table (each element's node indices). Cache the result on the `ZAccessor` instance — the underlying `_zn` is mutable-by-rebinding only through DataArray construction, so a one-shot cache keyed to the accessor's identity is sufficient. **Workaround if `_zn` later becomes mutable in-place**: invalidate on every access.
- **Avoid**: adding a public `ZAccessor` symbol to `mikeio.__init__`. **Instead**: keep it module-private; users only interact through `da.z` / `ds.z`.


## Implementation Plan

### Implementation Tasks

- [x] **TI01** A `ZAccessor` class lives in `src/mikeio/dataset/_z_accessor.py` and exposes read-only `nodes` and `elements` properties for layered DataArrays
  - Constructor takes the parent `DataArray`. `nodes` returns `da._zn` unchanged. `elements` lazily computes the mean of column-node z's per element using the geometry's element table and caches the result. Fully typed; no `Any`.
  - **Verify**: `Test: ZAccessor(da).nodes is shape (n_time, n_nodes) and equals da._zn; ZAccessor(da).elements is shape (n_time, n_elements) and equals the per-element node-mean computed independently from da.geometry.element_table; second call to .elements returns the same object`.

- [x] **TI02** A `NullZAccessor` class in the same module raises a clear `AttributeError` for any attribute access, naming the geometry type
  - Used for non-layered geometries. Error message format: `"DataArray with geometry '<ClassName>' has no z-coordinates; only layered 3D dfsu DataArrays expose .z"`.
  - **Verify**: `Test: NullZAccessor(Grid2D-backed-da).nodes raises AttributeError whose message contains 'Grid2D' and 'has no z-coordinates'`.

- [x] **TI03** `DataArray.z` is populated in `__init__` by a dispatcher mirroring `_get_plotter_by_geometry`
  - Add `self.z = self._get_z_accessor_by_geometry()` directly after [L171](../../src/mikeio/dataset/_dataarray.py#L171). Dispatcher returns `ZAccessor(self)` for `GeometryFM3D`, `GeometryFMVerticalProfile`, `GeometryFMVerticalColumn` (the geometries where `_zn` is populated); `NullZAccessor(self)` otherwise.
  - **Verify**: `Test: read a 3D layered dfsu, assert isinstance(da.z, ZAccessor); read a dfs2, assert isinstance(da.z, NullZAccessor)`.

- [x] **TI04** `Dataset.z` is a property returning `self[0].z`, mirroring the `_zn` delegation pattern
  - Add to `src/mikeio/dataset/_dataset.py` next to the existing `_zn` property at [L331](../../src/mikeio/dataset/_dataset.py#L331).
  - **Verify**: `Test: ds.z.nodes is ds[0].z.nodes (same array); ds.z.elements equals ds[0].z.elements`.

- [x] **TI05** Vertical-profile plotter reads z-coordinates via the public accessor
  - Replace `self.da._zn` at [`_data_plot.py:638`](../../src/mikeio/dataset/_data_plot.py#L638) with `self.da.z.nodes`. The downstream call at L714-720 that passes `zn=da._zn` to a helper should likewise switch.
  - **Verify**: `Test: existing vertical-profile plot tests still pass (no visual regression in tests/test_dfsu_plot.py or equivalent)`.

- [x] **TI06** 3D dfsu writer reads z-coordinates via the public accessor
  - Replace `_zn` read at [`_dfsu.py:125`](../../src/mikeio/dfsu/_dfsu.py#L125) with the public accessor. The writer-side coupling is internal; this is a rename for clarity. If the writer needs node-shape `(n_time, n_nodes)` specifically, use `da.z.nodes`.
  - **Verify**: `Test: round-trip write + read of a 3D layered dfsu produces identical zn values; existing writer tests pass`.

- [x] **TI07** Dfsu → netCDF export notebook reads z-coordinates via the public accessor
  - Update `notebooks/Dfsu - Export to netcdf.ipynb` cells that reference `ds._zn` to `ds.z.nodes`. Re-execute the notebook to ensure outputs still render.
  - **Verify**: `Test: grep -n '_zn' notebooks/Dfsu*.ipynb returns no matches; notebook executes end-to-end without error`.

- [x] **TI08** Existing layered-dfsu tests assert on the public accessor instead of `_zn`
  - Migrate `tests/test_dfsu3d.py` lines 183-194 and 605-607: replace `dscol1._zn` / `ds._zn` with `dscol1.z.nodes` / `ds.z.nodes`. Adds at least one new test asserting `AttributeError` on a non-layered DataArray (S04). Adds at least one test asserting element-center derivation against an independent computation (S02).
  - **Verify**: `Test: uv run pytest tests/test_dfsu3d.py passes; grep -n '\._zn\b' tests/ returns no matches outside files that test the alias still exists`.

- [x] **TI09** A user-guide section documents `da.z.nodes` / `da.z.elements` with a worked dz-weighted column-mean example
  - New section in `docs/user-guide/dfsu-3d.qmd` (or the closest existing layered-3D doc). Show: (1) accessing `nodes` and `elements`, (2) shape semantics, (3) a dz-weighted column mean snippet, (4) what happens on a non-layered DataArray.
  - **Verify**: `Test: cd docs && uv run quarto render path/to/page renders without error; rendered page references da.z.nodes and da.z.elements`.

### Testing Strategy
> Default test approach: per-task Verify lines + scenario tests scaffolded from Acceptance Scenarios. **Leave empty** when this is sufficient; fill only when the test approach is non-obvious.

### Validation
> Standard validation (build/test checks, code review, visual validation, and 1-pass remediation) is handled by exec-spec.

### Execution Contract
- TI01 + TI02 + TI03 must all land before TI05/TI06/TI07/TI08 — consumers and tests reference the public API.
- TI05–TI08 are independent of each other and can run in parallel after TI03.
- TI09 can run anytime after TI01 lands.


## Final Validation Checklist
> Acceptance Scenarios, Structural Criteria, and task Verify lines are the standard completion gates. **Leave empty** when these are sufficient.

- [x] `grep -rn '\._zn\b' src/ tests/ docs/ notebooks/` returns matches only inside `src/mikeio/dataset/` (the implementation owns the private alias).


## Implementation Observations

> _Managed by exec-spec post-implementation – append-only._

### Run: 2026-05-19 (feat/issue-976-z-accessor)

#### NOTICED BUT NOT TOUCHING
- `src/mikeio/dfsu/_layered.py:594` previously read `ds[0]._zn` for surface-elevation extraction; migrated to `ds[0].z.nodes` since it was outside `src/mikeio/dataset/` and the Final Validation Checklist required only `src/mikeio/dataset/` matches.
- `tests/test_dataset.py:87` previously asserted `ds1._zn is None` on a Grid1D dataset; migrated to `pytest.raises(AttributeError, match="has no z-coordinates")` exercising the NullZAccessor path.
- `tests/test_dataset.py:1204-1205` (concat dfsu3d) previously asserted on `ds._zn` directly; migrated to `ds.z.nodes`.

#### IMPLEMENTATION NOTES
- Dispatcher in `_get_z_accessor_by_geometry` adds a `self._zn is not None` guard (in addition to the geometry-type check) so that a layered DataArray constructed without `zn` falls through to `NullZAccessor` and raises a clean `AttributeError` from `.nodes`, rather than tripping an internal `assert`. This was driven by a correctness-review finding; the FIS parenthetical "(the geometries where `_zn` is populated)" supports this defensive branch.
- `NullZAccessor.__getattr__` skips dunder names with a plain `AttributeError(name)` so that deepcopy, IPython repr probes, etc. don't trigger the long user-facing message. Also looks up `_geometry_type_name` via `self.__dict__` to avoid `__getattr__` recursion when the instance has been reconstructed without `__init__`.
- Element-center cache is one-shot keyed on `is None?` per the FIS Constraints & Gotchas (one-shot cache accepted; in-place mutation of `_zn` is documented as an open footgun).

