---
title: "UGRID-Compliant NetCDF Export"
status: "Under Consideration"
category: "Interoperability"
summary: "Export dfsu mesh topology and data as UGRID-compliant NetCDF, readable by QGIS, ParaView, xugrid, and other standards-aware tools."
---

## Value Proposition

UGRID is the established convention for storing unstructured mesh data in NetCDF. It is supported by QGIS (mesh layer), ParaView, Deltares xugrid, MDAL, and many other tools. Currently, sharing dfsu results with collaborators requires them to install MIKE IO. A UGRID-compliant NetCDF export would make MIKE simulation output accessible to anyone with standard scientific software.

## What This Enables

- **QGIS visualization**: Open exported files directly as QGIS mesh layers — no plugins or converters needed
- **xugrid interoperability**: Deltares xugrid can read UGRID NetCDF natively, unlocking regridding, spatial analysis, and GeoDataFrame conversion
- **ParaView rendering**: 3D visualization of simulation results in ParaView via its UGRID reader
- **Archival and sharing**: NetCDF is a widely accepted archival format with rich metadata support
- **Cloud workflows**: NetCDF files can be served via THREDDS/OPeNDAP or converted to Zarr for cloud-native access

## Current State

`to_xarray()` converts data to xarray objects but does not encode mesh topology. The mesh structure (node coordinates, element-node connectivity, element types, boundary codes) is lost in the conversion. The existing `netcdf-zarr-conversion` roadmap item identifies this gap; this feature provides the specific approach.

## Design Considerations

- Target UGRID 1.0 conventions, which define `mesh_topology` variable with `cf_role`, `node_coordinates`, `face_node_connectivity`, and related attributes
- Map element-based data (dfsu type 2001) to face-located variables, face-based data (type 2004) to edge-located variables
- Include boundary codes as a node attribute (`boundary_node_connectivity` or via node `flag` variable)
- Handle mixed meshes (triangles + quads) via padded connectivity arrays with fill values, per UGRID spec
- Requires EUM-to-CF mapping for variable metadata (depends on the CF standard name mapping feature)
- 3D layered meshes need careful consideration — UGRID has limited 3D support; a practical approach may be to export per-layer or use the UGRID layered extension
- Projection information should be written as a CF grid mapping variable
