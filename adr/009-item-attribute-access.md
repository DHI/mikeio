# ADR-009: Item Attribute Access on Dataset

**Status:** Accepted — attribute access frozen; `ds["name"]` is canonical; full deprecation under evaluation for v4
**Date:** 2026-06

## Context

A `Dataset` is a name-keyed collection of `DataArray` items (see
[ADR-003](003-dataset-dataarray-pattern.md)). The canonical way to select an item
is by key: `ds["Surface elevation"]`.

Following the [pandas](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)
/ [xarray](https://docs.xarray.dev/en/stable/user-guide/data-structures.html#dictionary-like-methods)
convention that MIKE IO's data structures were modelled on, items are *also*
exposed as instance attributes for ergonomics and tab-completion:
`ds.Surface_elevation`. This is implemented in `Dataset._set_name_attr` by
`setattr(self, safe_name, da)`, where `safe_name` replaces non-alphanumeric
characters with underscores. The feature is currently *taught*: the user guide
lists `ds.itemA` as a selection method and demos it.

Two forces now point away from this design:

- **Familiarity has shifted.** Staying close to peer libraries is a deliberate
  MIKE IO design goal. When the reference set was pandas + xarray, attribute access
  was the familiar choice. But newer libraries — notably
  [polars](https://docs.pola.rs/) — deliberately dropped it: there is no
  `df.colname`, only `df["colname"]`/expressions, precisely to avoid the
  namespace-collision and implicit-magic costs. "Be familiar" now points *away* from
  attribute access for a post-2020 dataframe mental model.
- **It is incompatible with the type-hint commitment.** The codebase enforces
  `disallow_untyped_defs` under mypy, yet dynamically `setattr`-ed item attributes
  are invisible to the type checker — every attribute-access line in the tests
  carries `# type: ignore`. It is the one corner of the API that structurally cannot
  honour the discipline the rest of the codebase holds.

The cost is also concrete, not hypothetical: item names share a single namespace
with the entire public API, so an item whose safe-name matches a class member —
a property (`z`, `geometry`, `shape`, …) or a method (`mean`, `max`, …) — collides
with it. MIKE IO itself manufactures such names: `aggregate(axis="items")` names the
result after the aggregation function (`mean`, `nanmean`, `max`). And once `z` became
a setter-less property in [#977](https://github.com/DHI/mikeio/pull/977), the
`setattr` collision changed from *silently shadowing* the member to raising
`AttributeError`, which broke `mikeio.read()` on any dfs0 carrying a `z` item
([#982](https://github.com/DHI/mikeio/pull/982)).

## Decision

`ds["name"]` is the canonical item interface and the form documentation and examples
should use going forward. It is also strictly more faithful: it represents real item
names containing spaces (`ds["Surface elevation"]`), which the attribute form cannot
without mangling.

Attribute access is **frozen legacy**: still supported, but no new code or docs
should rely on it, and it is on a path to deprecation (`DeprecationWarning`) and
removal, targeted for v4.

Until then, collisions are made safe rather than fatal. When an item's safe-name
matches a class member, `Dataset._set_name_attr` **skips** setting the attribute;
`ds.<name>` resolves to the real property/method and the item remains accessible via
`ds["name"]`. The skip is **silent** — no warning, no error — because the collision
is part of normal usage (aggregation generates method-like names), so a warning would
fire on routine, correct code, and because it matches how `time`/`geometry` items
already behave.

## Alternatives Considered

- **Bless attribute access and keep it indefinitely**: rejected — contradicts both
  the type-hint direction and the polars-era familiarity argument above.
- **Warn on every collision**: rejected — `aggregate(axis="items")` produces names
  like `mean`/`nanmean` on a documented operation, so the warning would fire on
  common, correct code.
- **Raise on collision**: rejected — would make real files (dfs0 with a `z` item)
  unreadable.
- **Remove attribute access now**: rejected — too abrupt; needs a deprecation cycle
  and a doc migration first.

## Consequences

- `ds["name"]` is the reliable, recommended way to access items.
- For an item named like a class member, `ds.<name>` returns the member, not the
  item — the same footgun pandas/xarray carry, now made safe rather than fatal.
- The collision guard in `_set_name_attr`/`_del_name_attr` is transitional. The
  planned deprecation mechanism — drop the `setattr` loop and route item lookup
  through `__getattr__` (which fires only on failed normal lookup) — would make
  collisions structurally impossible *and* provide the natural site for a
  `DeprecationWarning` on use; the guard is removed when that lands.
- Deprecation will require a doc migration: rewrite the user-guide selection list and
  convert examples (`ds.elevation`, `ds.lon`/`ds.lat`, …) to `ds["..."]`. The
  "named items for readability" guidance is unaffected — it is a named-vs-positional
  recommendation that `ds["name"]` already satisfies.
- Removal is a breaking change and a v4 candidate; it would be recorded in a later
  ADR superseding this one.
