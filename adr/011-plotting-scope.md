# ADR-011: Plotting Is a Quick Look With One Way to Do Each Thing

**Status:** Accepted
**Date:** 2026-11

Amends [ADR-006](006-matplotlib-style-plotting.md).

## Context

ADR-006 added a `.plot` accessor so users can look at their data quickly. The plotting API has since grown in layers. A `plot_type=` keyword came first (2020, `8ec135fc`). Methods such as `.contourf()` were added later (2022, `ecb3f74f`) without retiring it. The result is several ways to do the same thing:

- `geometry.plot(plot_type="contourf")` and `geometry.plot.contourf()`;
- `patch` on flexible-mesh and spectrum plots, `pcolormesh` on grids, for the same "colour each cell" concept (the spectrum `patch` calls `ax.pcolormesh`);
- `cmin`/`cmax` on vertical profiles, `vmin`/`vmax` everywhere else;
- `rmin`/`rmax` duplicating `ax.set_rmin()`/`ax.set_rmax()`.

Grid plots always draw a colorbar, while FM plots accept `add_colorbar`. Signatures differ in argument order and in whether `title` is named (#969). The maintainer could not tell from the code which of `plot_type=` or the methods was the canonical way. If the person who maintains an API can't tell, its users can't either.

## Decision

MIKE IO plotting exists for a quick look at the data, in the style of xarray:

```python
da.isel(time=0).plot.contourf()
```

It is not meant to produce report-grade figures. Users customise the returned `Axes` with matplotlib.

**Rule.** One name per concept, one way to do each thing, the same signature on every geometry. Where xarray has the concept, use xarray's name. Maintenance cost breaks ties.

**Call contract.** Every plotting method has the signature

```python
def method(self, ax=None, *, figsize=None, title=None, label=None, add_colorbar=True, **kwargs) -> Axes
```

and returns the matplotlib `Axes`. `add_colorbar` appears only on methods that draw a colorbar. `title` and `label` default to MIKE metadata (item name, unit, timestep). All other `**kwargs` go to the underlying matplotlib call.

Flexible-mesh maps additionally name `cmap`, `vmin`, `vmax`, `levels`, `show_mesh` and `show_outline`. An FM map is several matplotlib calls (patches or tricontour, colorbar, optional mesh and outline overlays) that have to share one colour normalisation. `PatchCollection` has no `levels` argument, so plain passthrough can't do this. Spectrum plots additionally name `r_as_periods`, which needs knowledge of the spectral geometry.

**Scope of alignment.** Names, argument order and the shared aspect/label code are aligned across the existing plotters (#969). New plotters (for `Grid1D`, `Grid3D`, vertical columns, spectral geometries) and a `Plotter` Protocol are not added. They would grow the API, not streamline it.

**Dependency.** matplotlib remains a required dependency. Making it optional would add an import guard, a code path without matplotlib, and a CI job to test that path.

## Changes

Deprecated in 3.5 with `FutureWarning`, removed in 4.0:

| Item | Replacement |
|---|---|
| `plot_type=` keyword | the method of that name: `.contourf()`, `.mesh()`, `.outline()`, … |
| `plot_type="shaded"` | none; `ax.tripcolor(..., shading="gouraud")` |
| `.patch()` on FM and spectrum plots | `.pcolormesh()` |
| `cmin`, `cmax` | `vmin`, `vmax` |
| `rmin`, `rmax` on spectrum plots | `ax.set_rmin()`, `ax.set_rmax()` |
| `n_refinements` | none |
| `Dfsu2DV.plot_vertical_profile` | `DataArray.plot()` |

Added in 3.5:

- `.pcolormesh()` on FM and spectrum plots;
- `add_colorbar` on grid plots, which until now always drew one.

Changed in 4.0 without deprecation, since nothing raises a warning:

- arguments after `ax` become keyword-only;
- `GeometryFMPlotter.mesh` and `.outline` take `ax` first, like every other method;
- the hard-coded compass gridlines and the `::2` tick stride in polar spectrum plots are dropped.

## Alternatives Considered

- **Remove only what is little used and costly** (the first draft of this ADR): Keeps cheap duplicates such as `plot_type=` alongside the methods, which is what confuses users.
- **Keep only thin wrappers over a single matplotlib call**: Removes flexible-mesh plotting, which is the main reason MIKE IO has plotting at all.
- **Complete #969 including new plotters and a Protocol**: Makes the surface uniform by growing it.
- **matplotlib as an optional extra**: See *Dependency*.

## Consequences

- Each plot type has one name, and the same name on every geometry where it applies.
- `_plot_map` and the polar spectrum code get simpler in 4.0.
- Users who need publication-quality figures customise with matplotlib.
- Point-in-polygon tests in geometry code move off `matplotlib.path`, so geometry does not depend on a plotting library.
