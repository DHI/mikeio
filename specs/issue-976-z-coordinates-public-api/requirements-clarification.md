# Requirements Clarification: Public z-coordinate accessor for layered dfsu DataArrays

Refs [#976](https://github.com/DHI/mikeio/issues/976).

## Summary

Expose the per-timestep node z-coordinates of 3D layered dfsu data through a stable public API, replacing direct reliance on the private `DataArray._zn` / `Dataset._zn` attribute. The accessor is read-only and applies only to DataArrays derived from layered files (3D layered dfsu plus 2D slices that retain `_zn`). The *how* — where the API lives in the type hierarchy, what shape it returns, and what it is called — is deferred to a trade-off analysis (`andthen:architecture --mode trade-off`) because the data is meaningful only for a subset of geometries that share a single `DataArray` class.

## Scope

### In Scope
- A public, read-only way to obtain the time-varying node z-coordinates currently held in `DataArray._zn` / `Dataset._zn`.
- A separately-exposed element-center z-coordinate view derived from the node values.
- Coverage of every DataArray instance where `_zn` is currently populated — 3D layered dfsu and the 2D vertical-profile / column-slice DataArrays produced from them.
- Migration of in-tree consumers (vertical-profile plot at `src/mikeio/dataset/_data_plot.py:638`, 3D writer at `src/mikeio/dfsu/_dfsu.py:125`, the [Dfsu → netCDF export notebook](https://github.com/DHI/mikeio/blob/main/notebooks/Dfsu%20-%20Export%20to%20netcdf.ipynb)) onto the new public name once it is chosen.

### Out of Scope
- A writable / setter API. The accessor is read-only in this iteration; editing z-coordinates before writing a 3D dfsu remains an internal concern.
- A new time-varying geometry abstraction. The shared `DataArray.geometry` stays static; whether the accessor *re-uses* the geometry to wrap its output is a downstream decision.
- CF-conformant NetCDF export tooling. Easier CF export is cited as motivation but is not delivered here.
- Behaviour on non-layered DataArrays beyond a clear, predictable error path (specifics depend on the chosen scoping mechanism).

### MVP Boundary
Public read-only accessors for both node-based and element-center z-coordinates on every layered-derived DataArray and the parent Dataset, with an explicit, predictable failure mode on non-layered DataArrays. In-tree consumers migrated to the new name.

### Not Doing (for now)
- **Setter / write path.** Deferred — would need shape, monotonicity-per-column, and layer-count validation, none of which is required by the motivating use cases.
- **Time-varying geometry refactor.** Deferred — current geometries are immutable and shared across timesteps; reworking that is far beyond the scope of exposing already-stored data.
- **CF-compliant export helpers.** Deferred — out of scope; this work only makes such helpers easier to build.

## Functional Requirements

### User Stories
- As a user analysing 3D layered dfsu output, I want to obtain `z(t)` for every node alongside `T(t)`, `S(t)`, so that I can compute pycnocline depth, mixed-layer depth, and buoyancy frequency without touching a private attribute.
- As a user computing depth-averaged quantities, I want `dz(t)` per column from layer-interface elevations, so that I can produce volume-weighted means that respect free-surface motion.
- As a user exporting a 3D dfsu to netCDF, I want a stable public name for the dynamic z-coordinate, so that the [export notebook](https://github.com/DHI/mikeio/blob/main/notebooks/Dfsu%20-%20Export%20to%20netcdf.ipynb) does not depend on `ds._zn`.
- As a user re-gridding to fixed z-levels for comparison with CTD casts or moorings, I want element-center z's per timestep, so that I can interpolate without re-deriving them from nodes.

### Core Flow
1. User reads a 3D layered dfsu: `ds = mikeio.read("layered.dfsu")`.
2. User obtains node z's per timestep through the public accessor (exact spelling TBD), shape `(n_time, n_nodes)`.
3. User obtains element-center z's per timestep through a sibling accessor, shape `(n_time, n_elements)`.
4. Both accessors return data aligned to `ds.time`.

### Alternate Flow
- User has a vertical-profile / column-slice DataArray (produced by `_dataarray.py:707-709` and similar paths). The same accessors return the appropriate node / element subset; shape conventions are part of the deferred design.

## Edge Cases

| Scenario | Expected Behavior |
|---|---|
| Non-layered DataArray (dfs0–dfs3, dfsu 2DH, spectral) | Predictable failure (exact mechanism depends on scoping choice). Never silently returns `None`. |
| Vertical-profile / 2D slice DataArray with populated `_zn` | Accessor returns the slice-appropriate z's; shape conventions TBD with the return-type decision. |
| Single-timestep selection (`isel(time=0)`) | Returns z's for that timestep, consistent with how `_zn` is currently subset at `_dataarray.py:689`. |
| `Dataset.z_coordinates` vs `DataArray.z_coordinates` | Dataset-level accessor returns the shared array (it is identical across items in a Dataset by construction; see `_dataset.py:331`). |

## Non-Functional Requirements
- **Backwards compatibility.** `_zn` may continue to exist internally for one release cycle to avoid churning the writer and plot paths in the same PR; the public name is the supported entry point thereafter.
- **No extra I/O cost.** The data is already loaded; the accessor must not re-read or copy unnecessarily. Element-center derivation may cache its result.
- **No data-dependent typing.** The presence or absence of the attribute must be predictable from the DataArray's type / origin, not from a runtime check on its data — see `feedback_no_data_dependent_api_behaviour`.

## Success Criteria
- [ ] Layered-derived DataArrays expose a public read-only way to retrieve `(n_time, n_nodes)` node z's.
- [ ] Layered-derived DataArrays expose a public read-only way to retrieve `(n_time, n_elements)` element-center z's.
- [ ] Non-layered DataArrays fail predictably and discoverably when the accessor is used.
- [ ] In-tree consumers (`_data_plot.py:638`, `_dfsu.py:125`, the export notebook) reference the public name.
- [ ] Documentation in the user guide explains the accessor, its shape, and at least one motivating example (e.g. dz-weighted column mean).

## Dependencies

| Dependency | Purpose | Risk |
|---|---|---|
| `andthen:architecture --mode trade-off` outcome | Resolves return-type, scoping mechanism, and naming | Spec / implementation cannot start until the trade-off lands. |

## Open Design Questions

These are deferred to [`andthen:architecture --mode trade-off`](https://github.com/IT-HUSET/andthen) and must be resolved before spec / implementation:

1. **Return type.** Raw `np.ndarray` of shape `(n_time, n_nodes)` vs. an `xarray.DataArray`-like wrapper carrying `time` and `node` dims vs. a new time-varying geometry wrapper (most ambitious).
2. **Scoping mechanism — how a layered-only API lives on a class shared across all geometries.** Candidates: free function (`mikeio.z_coordinates(da)`), layered-specific DataArray subclass returned by the existing factory, pandas-style accessor namespace (`da.z.nodes` / `da.z.elements`), or another option uncovered during the trade-off. This is the postponed decision that motivated opening [#976](https://github.com/DHI/mikeio/issues/976).
3. **Naming.** Depends on the scoping mechanism — verb-y (`z_coordinates(da)`) for a function, noun-y (`.z_coordinates_nodes` / `.z_coordinates_elements` or `.node_z` / `.element_z`) for an attribute, terse (`.z.n` / `.z.e`) for a namespace. José Antonio Arenas suggested `z_coordinates` as the stem.

## Decisions Log

| Decision | Rationale | Date |
|---|---|---|
| Expose both node-based and element-center z-coordinates as separate views | José Antonio Arenas ([#976 comment](https://github.com/DHI/mikeio/issues/976#issuecomment-)) asked for both; separating them keeps each access pattern direct, and element-center derivation can be cached without entangling node access. | 2026-05-19 |
| Read-only in this iteration; no setter | Smallest viable surface; writer-side coupling to `_zn` is an internal concern and a setter would need validation work not motivated by the use cases listed. | 2026-05-19 |
| Apply to every DataArray that currently carries `_zn`, not just whole-file 3D layered DataArrays | Avoids the surprise where a vertical slice silently drops the accessor; matches the internal invariant that `_zn` is populated wherever it is meaningful. | 2026-05-19 |
| Defer return-type, scoping mechanism, and naming to `andthen:architecture --mode trade-off` | The maintainer postponed [#976](https://github.com/DHI/mikeio/issues/976) precisely because exposing a layered-only attribute on the shared `DataArray` class is non-trivial; a formal trade-off should weigh subclass refactor cost vs. free-function ergonomics vs. namespace discoverability before clarify resumes. | 2026-05-19 |
