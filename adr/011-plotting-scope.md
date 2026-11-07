# ADR-011: Plotting Is for a Quick Look

**Status:** Accepted
**Date:** 2026-11

Amends [ADR-006](006-matplotlib-style-plotting.md).

## Context

ADR-006 added a `.plot` accessor so users can look at their data quickly. Since then the plotting code has grown options aimed at polished output: mesh refinement, colorbar toggles, outline toggles, polar-plot layout. That code accounts for a large share of open plotting issues (#914, #1017, #1024, #1025), and the maintainer's time is limited.

## Decision

MIKE IO plotting exists for a quick look at the data, in the style of xarray:

```python
da.isel(time=0).plot.contourf()
```

It is not meant to produce report-grade figures for every use case. Users customise the returned `Axes` with matplotlib.

**Call contract.** Every plotting method:

- accepts `ax=` and returns the matplotlib `Axes`;
- has `title`, `label`, `figsize` and `ax` as its only named kwargs, because their defaults come from MIKE metadata (item name, unit, timestep);
- passes all other `**kwargs` to the underlying matplotlib call.

The one exception is flexible-mesh maps (`plot`, `patch`, `contour`, `contourf` on FM geometries). They also name `cmap`, `vmin`, `vmax`, `levels` and `show_mesh`. An FM map is several matplotlib calls (patches or tricontour, colorbar, optional mesh overlay) that have to share one colour normalisation. `PatchCollection` has no `levels` argument, so plain passthrough can't do this.

**Removal rule.** A plotting feature is removed when it is both little used and costly to maintain. Usage evidence is the documentation and the issue tracker. A feature that is unused but implemented as a thin wrapper over one matplotlib call stays. Removing it would break users and save almost no code.

**Dependency.** matplotlib remains a required dependency. Making it optional would add an import guard, a code path without matplotlib, and a CI job to test that path. That is more maintenance, not less.

## Changes

Deprecated in 3.5 with `FutureWarning`, removed in 4.0:

| Item | Replacement |
|---|---|
| `n_refinements` | none |
| `show_outline` | `da.geometry.plot.outline(ax=ax)` |
| `add_colorbar=False` | remove the colorbar axes: `ax.figure.axes[-1].remove()` |
| `cmin`, `cmax` | `vmin`, `vmax` |
| `rmin` (spectrum) | `ax.set_rmin(...)` |
| `Dfsu2DV.plot_vertical_profile` | `DataArray.plot()` |

Changed in 4.0 without deprecation, since nothing raises: the hard-coded compass gridlines and the `::2` tick stride in polar spectrum plots are dropped.

Every plotting method stays, along with `rmax` and `r_as_periods`.

## Alternatives Considered

- **Keep only thin wrappers over a single matplotlib call**: Removes flexible-mesh plotting, which is the main reason MIKE IO has plotting at all.
- **Remove everything the docs never use** (`imshow`, `pcolormesh`, geometry `contour`/`contourf`, `boundary_nodes`): `pcolormesh` is the default for `Grid2D.plot()`, and the others are thin. Removing them breaks users and saves little.
- **matplotlib as an optional extra**: Discussed under *Dependency* above.

## Consequences

- `_plot_map` and the polar spectrum code get simpler in 4.0. Most of the maintenance burden sits there, not in the number of methods.
- Users who need publication-quality figures customise with matplotlib.
- Point-in-polygon tests in geometry code move off `matplotlib.path`. matplotlib stays required, but geometry should not depend on a plotting library.
