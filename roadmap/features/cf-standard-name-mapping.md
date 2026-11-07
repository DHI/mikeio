---
title: "EUM to CF Standard Name Mapping"
status: "Under Consideration"
category: "Interoperability"
summary: "Map DHI EUM types and units to CF convention standard names and UDUNITS, enabling standards-compliant metadata in NetCDF, Zarr, and xarray exports."
---

## Value Proposition

The CF (Climate and Forecast) conventions are the lingua franca of earth science data. Tools like xarray, QGIS, ParaView, CDO, NCO, and cf-xarray all rely on `standard_name` and `units` attributes to interpret variables correctly. MIKE IO's EUM type system is rich (hundreds of physical quantities with associated units), but it is proprietary — no external tool understands `eumType=100006`. A mapping table would make every export from MIKE IO immediately meaningful to the broader ecosystem.

## What This Enables

- **Standards-compliant xarray export**: `to_xarray()` populates `standard_name`, `long_name`, and `units` attributes automatically
- **Foundation for UGRID and NetCDF export**: CF metadata is a prerequisite for valid UGRID-compliant NetCDF files
- **Tool interoperability**: cf-xarray, Iris, CDO, NCO, and CMIP tooling can discover and operate on variables by standard name
- **Unit consistency**: Map EUM units to UDUNITS-compatible strings (e.g., `meter_per_second` rather than `m/s per sqrt(Hz)`)

## Current State

`to_xarray()` sets `name`, `units`, `eumType`, and `eumUnit` attributes from `ItemInfo`. The `units` string comes from the EUM system and is human-readable but not UDUNITS-compliant. No `standard_name` or `long_name` is set.

## Design Considerations

- The mapping should cover the most common EUM types used in MIKE 21/3 output (water level, velocity, salinity, temperature, wave parameters, etc.) rather than attempting to map all ~1000 EUM types
- Some EUM types have no CF equivalent (e.g., MIKE-specific calibration parameters) — these should fall back to `long_name` only
- The mapping table should be maintainable as a simple data structure (dict or CSV), not scattered across code
- Unit strings must be UDUNITS-2 compatible (e.g., `m s-1` not `m/s`)
