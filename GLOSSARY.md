# MIKE IO

Python library for reading, writing, and manipulating MIKE files (dfs0–dfs3, dfsu, mesh) from [DHI](https://www.dhigroup.com/) (Danish Hydraulic Institute). Sits between the DHI/MIKE file-format world and the Python scientific stack (numpy, pandas, xarray).

## Who this is for

This document defines the vocabulary used across mikeio's code, docs, and discussions. Read it if you're:

- **using mikeio** and want to understand the concepts behind the API
- **contributing code or docs** and want to use names consistently
- an **LLM/agent** working in this repo

If you only need to use mikeio, the [user guide](docs/user-guide/) and [API reference](docs/api/) are the right entry points; come here when a term is unfamiliar.

## Mental model

A dfs file on disk holds one or more **Items** (named quantities with metadata) sampled along a **Time axis** over some **Geometry** (a structured **Grid** or a **Flexible Mesh**). Reading produces an in-memory **Dataset** of **DataArrays** — one DataArray per Item — all sharing the same time axis and geometry.

```text
file on disk         in memory
────────────         ─────────
dfs / dfsu  ──read──▶ Dataset
  └─ Item             ├─ DataArray
     ├─ ItemInfo      │   ├─ ItemInfo
     └─ values        │   ├─ values
                      ├─ Time axis
                      └─ Geometry
```

The rest of this document defines each named box and the variants you may encounter.

## Capitalisation

DHI's own docs use **DFS0/DFS1/DFS2/DFS3/DFSU** (uppercase). mikeio uses lowercase `dfs0`–`dfs3`/`dfsu` for prose and file extensions, and PascalCase (`Dfs0`, `Dfsu2DH`, …) for Python classes. Both are accepted; uppercase is closer to DHI, lowercase matches mikeio convention.

## Language

### File formats

**DFS** (uppercase):
The DHI binary file-format family ("Data File System"). Refers to the format itself and its specification.
_Avoid_: dfs (when meant as the format spec), MIKE format.

**dfs file** (lowercase):
A file in the DFS family — includes `.dfs0`, `.dfs1`, `.dfs2`, `.dfs3`, and `.dfsu`. Does **not** include `.mesh`.
_Avoid_: MIKE file (too broad), DHI file.

**dfs0 / dfs1 / dfs2 / dfs3**:
Structured-grid dfs files. The digit is the spatial dimensionality: 0 = time series (point), 1 = line, 2 = area, 3 = volume. All carry a time axis even when single-snapshot.
_Avoid_: 0D/1D/2D/3D dfs (informal — use dfs0..dfs3), dfs-N.

**dfsu**:
Flexible-mesh dfs file. The "u" stands for unstructured. Subtypes are 2DH (horizontal), 2DV (vertical slice), 3D (layered), and Spectral.
_Avoid_: unstructured dfs, FM dfs.

**Mesh file**:
A `.mesh` file containing only mesh topology (Nodes + Elements) with no dynamic data. Distinct from dfsu.
_Avoid_: mesh-only dfs (it's not a dfs file).

### Spatial primitives

**Element**:
A discrete area/volume in a flexible mesh (unstructured grid). Triangles and quads in 2D; prisms and hexahedra in 3D. Note: DHI's own docs sometimes use "Element" loosely to describe value location for *structured* grids too — mikeio does not, see **Cell** below.
_Avoid_: cell, face, polygon (in FM context).

**Cell**:
A discrete area/volume in a structured grid (`Grid1D`, `Grid2D`, `Grid3D`).
_Avoid_: element, pixel (in Grid context).

**Node**:
A mesh vertex — a point with coordinates referenced by Element connectivity.
_Avoid_: vertex, point, corner.

**Face**:
The interface between an Element and one neighbouring Element, or between an Element and the domain boundary. Carries data for DFSU type 2004 (e.g. discharge across an interface). In a **Locally refined mesh**, one coarse Face can meet two or more fine Faces at **Hanging nodes**. In layered 3D meshes, "layer face" is the horizontal interface between two stacked Elements.
_Avoid_: edge, interface, side.

**Hanging node**:
A Node that lies on a Face of a neighbouring Element but is not one of that Element's Nodes.
_Avoid_: dangling node, T-junction, irregular node.

**Locally refined mesh**:
A Flexible Mesh that contains Hanging nodes, where finer Elements meet coarser ones.
_Avoid_: submesh 1, non-conforming mesh, refined submesh.

### Spatial structures

**Geometry**:
The spatial structure attached to a DataArray/Dataset. Abstract concept; concrete types are `Grid1D`, `Grid2D`, `Grid3D`, `GeometryFM2D`, `GeometryFM3D`, `GeometryFMVerticalProfile`, etc.
_Avoid_: grid (when meant generically), topology, domain.

**Grid**:
A structured (regular, axis-aligned) Geometry with uniform spacing — `Grid1D`, `Grid2D`, `Grid3D`. Backs dfs1/dfs2/dfs3.
_Avoid_: raster, lattice, mesh.

**Flexible Mesh (FM)**:
An unstructured Geometry made of Nodes and Elements — `GeometryFM2D`, `GeometryFM3D`, etc. Backs dfsu and `.mesh` files. The codebase uses `FM` consistently in class names.
_Avoid_: unstructured grid, triangulation, "mesh" (when meant as a geometry concept — see below).

**Mesh** (file):
A `.mesh` file: mesh topology only, no dynamic data. Reserved for the *file type*, never for the geometry concept.
_Avoid_: using "Mesh" for the unstructured geometry concept — say "Flexible Mesh" or "FM" instead.

**Projection**:
The coordinate-system reference of a dfs file's spatial axis (e.g. `LONG/LAT`, UTM zone, custom). Held alongside Geometry.
_Avoid_: CRS (the code uses `crs` utilities but file-side concept is "projection"), coordinate system.

### Engineering Unit Manager (EUM)

**EUM** ("Engineering Unit Manager"):
DHI's catalogue of physical quantities and their permissible units. Exposed in mikeio via `EUMType` (the kind, e.g. `Water_Level`) and `EUMUnit` (the unit, e.g. `meter`).
_Avoid_: unit system, quantity registry.

**EUMType**:
The physical-quantity kind of an Item (e.g. `Water_Level`, `Wind_speed`, `Discharge`). Constrains which `EUMUnit` values are valid.
_Avoid_: quantity, kind, item type.

**EUMUnit**:
The unit of measurement for an Item (e.g. `meter`, `meter_per_sec`). Must be compatible with the Item's `EUMType`.
_Avoid_: unit, units, uom.

**ItemInfo**:
The Python class bundling the metadata of one Item: `name`, `EUMType`, `EUMUnit`, `DataValueType`. When referring to the *concept* (e.g. in prose), prefer "Item metadata"; when referring to the class/object, use `ItemInfo`.

```python
>>> from mikeio import ItemInfo, EUMType, EUMUnit
>>> ItemInfo("Water Level", EUMType.Water_Level, EUMUnit.meter)
```

_Avoid_: item descriptor, item header, item spec.

**DataValueType**:
How a value relates to the time axis: `Instantaneous`, `Accumulated`, `StepAccumulated`, `MeanStepBackward`, `MeanStepForward`. Only meaningful for **dfs0**; dfs1/dfs2/dfs3 are always `Instantaneous`. Roughly analogous to [CF Conventions'](https://cf-convention.github.io/) `cell_methods` along the time axis (`time: point`, `time: sum`, `time: mean`), but DHI does not formalise the mapping.
_Avoid_: cell method, sampling type, temporal aggregation.

### Data on disk vs. in memory

**Item**:
A named quantity stored in a dfs file, carrying an `ItemInfo` (name, `EUMType`, `EUMUnit`). File-side concept.
_Avoid_: variable, column, channel, field.

**DataArray**:
The in-memory representation of one Item's data after reading, with its time axis, geometry, and ItemInfo attached.
_Avoid_: variable, array, series.

**Dataset**:
An ordered collection of DataArrays sharing the same time axis and geometry — the in-memory result of reading a dfs file.

```python
>>> import mikeio
>>> ds = mikeio.read("oresund.dfsu")
>>> ds["Surface elevation"]   # → DataArray
```

_Avoid_: frame, table.

### Time

**Time axis**:
The sequence of timestamps a Dataset/DataArray is indexed by — a `pandas.DatetimeIndex`. Always present in dfs files, even single-snapshot ones.
_Avoid_: time dimension, time coord (when meant as the axis itself).

**Timestep**:
Used in two senses, disambiguated by context:
1. **The interval** between consecutive samples (a duration in seconds — e.g. the `.timestep` property on a reader).
2. **A single sample** along the time axis (e.g. `n_timesteps`, "read timestep 0").
When ambiguity matters, write "timestep duration" vs "timestep index" / "snapshot".
_Avoid_: frame, record, tick.

**Snapshot**:
A single sample along the time axis — unambiguous synonym for the second sense of "timestep". Useful when discussing single-time-step files.
_Avoid_: frame, slice (which has its own meaning).

**Equidistant**:
A time axis with uniform spacing between samples. DHI calls this an "equidistant calendar axis". Non-equidistant axes exist and require reading the data to obtain the timestamps.
_Avoid_: uniform, regular (when meant as time-spacing).

### Parameter files (PFS)

**PFS file**:
A DHI text-format file containing parameters and settings for MIKE tools and engines. Distinct subdomain from dfs/dfsu data files.
_Avoid_: parameter file (too generic), config file, ini file.

**Target**:
A root (out-most) Section in a PFS file. A PFS file contains one or more Targets.
_Avoid_: root section (use "Target"), top-level block.

**Section**:
A named scope inside a PFS file. May contain nested Sections and Keywords. Represented in mikeio by `PfsSection` (a single class for any nesting depth).
_Avoid_: sub-section as a type (it's just a Section that happens to be nested), block, group.

**Keyword**:
A named entry inside a Section. Holds one or more Parameters.
_Avoid_: key, field, attribute.

**Parameter** (PFS context):
A value held by a Keyword. A Keyword can hold multiple Parameters (e.g. `point = 1, 2.5, 'name'` is one Keyword with three Parameters). Types: Integer, Double, Boolean, Text string, File name, CLOB.
_Avoid_: param, value (in PFS prose), entry. Note: "Parameter" in this PFS sense is **not** a key-value pair — that's a Keyword with its Parameters.

**PfsDocument**:
The Python class representing a parsed PFS file. No direct DHI counterpart name.
_Avoid_: pfs object, parsed pfs.

### Spectral data

**Spectrum** (noun):
A data distribution over a frequency axis, a direction axis, or both — typically wave energy density. The *data* held at one location.
_Avoid_: spectral data (when meant as a thing rather than a property).

**Spectral** (adjective):
Modifier for files, geometries, and items that carry Spectra (e.g. *spectral file*, *spectral geometry*).
_Avoid_: using "spectral" as a noun.

**Point / Line / Area spectrum**:
The spatial arrangement of where Spectra live in a spectral dfsu — a single point (`GeometryFMPointSpectrum`), along a 1D line (`GeometryFMLineSpectrum`), or over a 2D mesh (`GeometryFMAreaSpectrum`).
_Avoid_: 0D/1D/2D spectrum.

**Frequency axis**:
The frequency dimension of a Spectrum (Hz). May be equidistant or logarithmic.
_Avoid_: freq axis (in prose).

**Direction axis**:
The directional dimension of a Spectrum (degrees). Wave heading convention (from/to) is determined by the file.
_Avoid_: angle axis, theta axis.

**Spectral file**:
A dfsu storing Spectra (rather than scalar fields per Element). Class: `DfsuSpectral`.
_Avoid_: spectrum file (singular noun); wave file.

### Boundaries

**Code** (node code):
An integer attached to each Node in a Flexible Mesh. Used to classify boundary membership.
_Avoid_: flag, tag, marker.

**Boundary code**:
A Code > 0 — i.e. a Node lies on the mesh boundary. Internal Nodes have code 0.
_Avoid_: boundary marker, boundary flag.

**Land boundary**:
Boundary code == 1. The closed/no-flux boundary of the mesh.
_Avoid_: closed boundary, wall.

**Open boundary**:
Boundary code > 1. The flow-through/forced boundary; multiple open boundaries are distinguished by distinct codes (2, 3, …).
_Avoid_: forced boundary, inlet/outlet.

**Internal node**:
A Node with code == 0 — strictly inside the mesh, not on any boundary.
_Avoid_: interior node.

### Sampling patterns

**Track**:
A moving observation — a `(time, x, y[, z])` sequence (e.g. ship, buoy, satellite path). Has a time axis. Sampled from a dfsu via `extract_track`.
_Avoid_: trajectory, path, route.

**Transect**:
A fixed cross-section through space (no time). Used for vertical-profile plots and similar 2D slices through a 3D mesh.
_Avoid_: section, slice, cut.

### Layered 3D meshes

**Layer**:
A horizontal slab of Elements in a 3D Flexible Mesh. Stacked bottom-up; Elements in the same vertical column have consecutive indices.
_Avoid_: level (CF/oceanography term, not used in mikeio), slice.

**Sigma layer**:
A terrain-following layer (top portion of the water column). Sigma layers follow the free surface; their thickness varies in time.
_Avoid_: terrain-following layer, surface layer.

**z-layer** (hyphenated):
A fixed-elevation layer (bottom portion). z-layers have constant Z; the count varies with depth (`n_z_layers` is a maximum).
_Avoid_: fixed layer, depth layer, z layer (no hyphen).

**Sigma-z**:
The hybrid layering mode where a 3D file mixes sigma layers (above) with z-layers (below). When `n_sigma_layers < n_layers`, the file is sigma-z.
_Avoid_: hybrid layers, mixed layers.

**Column** (vertical):
The vertical stack of Elements above a single horizontal (x, y) position — represented by `GeometryFMVerticalColumn`.
_Avoid_: water column (overloaded — physical concept), stack.

**Vertical profile**:
A 2D vertical slice through a 3D mesh (a transect with depth). Backed by `Dfsu2DV` files and `GeometryFMVerticalProfile`.
_Avoid_: vertical transect, vertical section.

### File format internals

The full binary specification lives in [`docs/FM_FileSpecification.pdf`](docs/FM_FileSpecification.pdf). The two terms below leak into user-facing docstrings, so they are defined here.

**Static item**:
Time-independent metadata stored as an "item" in the dfs binary structure (e.g. node coordinates, element table, projection). Distinct from the user-visible **Item** concept above.
_Avoid_: metadata item, header item, fixed item.

**Dynamic item**:
A time-varying Item — what users mean when they say "Item" in everyday usage. The Item entry in the file's data block.
_Avoid_: data item, time-varying item.

**Submesh type**:
A value in the dfsu header, set by MIKE, that describes the mesh (e.g. 1 = Locally refined mesh). Files from before MIKE 2027 have none (`None` in MIKE IO). It is informative only: the Geometry itself is the source of truth.
_Avoid_: submesh flag, mesh type.

## Relationships

- A dfs file contains one or more **Items**; reading produces a **Dataset** of **DataArrays**, one per Item.
- A **DataArray** has exactly one **ItemInfo**; a **Dataset** has one **ItemInfo** per **DataArray**.
- A **Flexible Mesh** has **Nodes** and **Elements**; an Element references its Nodes via the element table (counter-clockwise).
- A **Grid** has **Cells** (no Nodes/Elements/Faces).
- A **DataArray** holds exactly one **Geometry**; a **Dataset** shares one Geometry across all its DataArrays.

## Example dialogue

> **User:** "I've got a dfsu file with surface elevation and wind speed. How do I get just the wind speed at a specific location?"
> **Maintainer:** "Open the file with `mikeio.read()` — that gives you a **Dataset** with two **DataArrays**, one per **Item**. Pick the wind-speed DataArray and use spatial selection on its **Geometry** (a `GeometryFM2D`) to get the **Element** containing your point."
>
> **User:** "And the surface-elevation values — are those at the corners or in the middle of the elements?"
> **Maintainer:** "**Element-centered**. dfsu data is always per-Element, never per-Node. mikeio has helpers to interpolate from element centres to **Nodes** when you need that."
>
> **User:** "Got it. Now I have a 3D file. What's the difference between sigma layers and z-layers?"
> **Maintainer:** "**Sigma layers** are terrain-following — they sit at the top of the water column and stretch with the free surface. **z-layers** are fixed-elevation, at the bottom. A file mixing both is **sigma-z**. The vertical stack at one (x, y) is a **Column**."
