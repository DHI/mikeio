# ADR-014: Hanging Nodes Are Found From the Geometry, Not the Submesh Type

**Status:** Accepted
**Date:** 2026-11

## Context

From MIKE 2027 a dfsu can hold a Locally refined mesh: fine Elements meet coarse Elements, and Hanging nodes lie on coarse Faces. The element table is unchanged. It stores plain triangles and quads, so a coarse Face and the fine Faces beside it each occur once, like Faces on the domain boundary. Boundary detection, the plot triangulation and node-centred data all expect each interior Face to be shared by exactly two Elements, so all three were wrong on these files (#1112).

MIKE writes a Submesh type to the dfsu header, 1 for a Locally refined mesh. It is the obvious switch for a fix, but it is a statement *about* the geometry and can be wrong:

- `to_dfs` and other tools have dropped it, so a Locally refined mesh can arrive with no Submesh type;
- after `sel` or `isel` it can still say 1 when no Hanging nodes are left;
- a geometry built in code has none.

## Decision

Hanging nodes are found from the node coordinates and the element table only. No code branches on the Submesh type.

Only Faces that occur once are candidates. A Node of another once-only Face is a Hanging node on Face A–C when it lies on segment A–C, strictly between A and C, within a tolerance relative to the Face length. Each Hanging node is inserted, in counter-clockwise order, as an extra corner of the Element that owns the Face. The result is an internal, cached *conforming element table* on `GeometryFM2D`, used for boundary Faces, the plot triangulation and node-centred data. `GeometryFM3D` gets it through its 2D geometry.

`element_table` stays exactly as stored in the file. The conforming element table is never exposed.

The Submesh type is read, kept on the geometry, and written back by `to_dfs` as read. Nothing else uses it.

## Alternatives Considered

**Branch on the Submesh type** — cheaper for files without Hanging nodes, but wrong for exactly the files that have lost the value, and the search is cheap anyway: a mesh with no candidates returns at once, and 320,000 triangles take about 0.2 s, once per geometry.

**Change `element_table` to the conforming table** — every consumer would be fixed at once, but the element table would no longer match the file, MIKE tools or the file specification, and writing it back would change the Element types in the file.

## Consequences

- Boundaries, outlines, contour plots and node values are correct for a Locally refined mesh whatever its header says, including subsets and geometries built in code.
- A file without Hanging nodes takes the old code paths: the conforming element table is then the same object as `element_table`.
- Code that needs Faces or neighbours from the element table should use the conforming element table.
- A subset written with `to_dfs` keeps Submesh type 1 even if it has no Hanging nodes left. This is harmless because nothing decides from the value.
