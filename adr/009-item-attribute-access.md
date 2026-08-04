# ADR-009: Item Attribute Access on Dataset

**Status:** Accepted — frozen legacy, deprecation targeted for v4
**Date:** 2026-06

## Context

A `Dataset` is a name-keyed collection of `DataArray` items (see
[ADR-003](003-dataset-dataarray-pattern.md)). The canonical way to select an item
is by key: `ds["Surface elevation"]`.

Following the [pandas](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)
/ [xarray](https://docs.xarray.dev/en/stable/user-guide/data-structures.html#dictionary-like-methods)
convention that MIKE IO's data structures were modelled on, items are also exposed as
instance attributes for ergonomics and tab-completion: `ds.Surface_elevation`. This is
implemented in `Dataset._set_name_attr` by `setattr(self, safe_name, da)`, where
`safe_name` replaces non-alphanumeric characters with underscores. The feature is
currently taught: the user guide lists `ds.itemA` as a selection method and demos it.

Two forces now point away from this design. The decisive one is the type-hint
commitment; a shift in the dataframe ecosystem corroborates it.

- **Incompatible with the type-hint commitment.** MIKE IO prefers explicitly-typed,
  key-based APIs and enforces `disallow_untyped_defs` under mypy. Dynamically
  `setattr`-ed item attributes are invisible to the type checker, so every
  attribute-access line in the tests carries a `# type: ignore`. It is the one corner
  of the API that cannot honour the discipline the rest of the codebase holds. Where
  the typed-API commitment conflicts with the pandas/xarray convention MIKE IO was
  modelled on, the commitment wins.
- **The ecosystem has moved the same way.** This is not a lone deviation from peer
  libraries. Newer dataframe libraries, notably [polars](https://docs.pola.rs/),
  dropped attribute access altogether: there is no `df.colname`, only
  `df["colname"]` and expressions, to avoid the namespace-collision and implicit-magic
  costs. Explicit key-based access is the post-2020 norm, so deviating from
  pandas/xarray here aligns MIKE IO with where the field has settled.

The cost is concrete, not hypothetical. Item names share a single namespace with the
entire public API, so an item whose safe-name matches a class member — a property
(`z`, `geometry`, `shape`) or a method (`mean`, `max`) — collides with it. MIKE IO
itself manufactures such names: `aggregate(axis="items")` names the result after the
aggregation function (`mean`, `nanmean`, `max`). Once `z` became a setter-less property
in [#977](https://github.com/DHI/mikeio/pull/977), the `setattr` collision changed from
silently shadowing the member to raising `AttributeError`, which broke `mikeio.read()`
on any dfs0 carrying a `z` item ([#982](https://github.com/DHI/mikeio/pull/982)).

## Decision

`ds["name"]` is the canonical item interface and the form documentation and examples
should use going forward. It is also more faithful: it represents real item names
containing spaces (`ds["Surface elevation"]`), which the attribute form cannot without
mangling.

Attribute access is frozen legacy: still supported, but no new code or docs should rely
on it, and it is on a path to deprecation (`DeprecationWarning`) and removal, targeted
for v4.

Until then, collisions are made safe rather than fatal. When an item's safe-name
matches a class member (property or method), or one of the instance attributes set in
`__init__` (`plot`, `title`, `_data_vars`, which are not class members and so are
reserved explicitly), `Dataset._set_name_attr` skips setting the attribute. `ds.<name>`
resolves to the real property, method, or accessor, and the item remains accessible via
`ds["name"]`. The skip is silent: no warning, no error. A warning would fire on routine,
correct code, since aggregation generates method-like names, and silence matches how
`time` and `geometry` items already behave.

## Alternatives Considered

- **Bless attribute access and keep it indefinitely**: rejected. Contradicts both the
  type-hint direction and the ecosystem shift above.
- **Warn on every collision**: rejected. `aggregate(axis="items")` produces names like
  `mean`/`nanmean` on a documented operation, so the warning would fire on common,
  correct code.
- **Raise on collision**: rejected. Would make real files (dfs0 with a `z` item)
  unreadable.
- **Remove attribute access now**: rejected. Too abrupt; needs a deprecation cycle and
  a doc migration first.

## Consequences

- `ds["name"]` is the reliable, recommended way to access items.
- For an item named like a class member, `ds.<name>` returns the member, not the item —
  the same footgun pandas/xarray carry, now made safe rather than fatal.
- The collision guard in `_set_name_attr`/`_del_name_attr` is transitional. The planned
  deprecation mechanism — drop the `setattr` loop and route item lookup through
  `__getattr__`, which fires only on failed normal lookup — would make collisions
  structurally impossible and provide the natural site for a `DeprecationWarning` on
  use. The guard is removed when that lands.
- Deprecation will require a doc migration: rewrite the user-guide selection list and
  convert examples (`ds.elevation`, `ds.lon`/`ds.lat`) to `ds["..."]`. The "named items
  for readability" guidance is unaffected — it is a named-vs-positional recommendation
  that `ds["name"]` already satisfies.
- Removal is a breaking change and a v4 candidate; it would be recorded in a later ADR
  superseding this one.
