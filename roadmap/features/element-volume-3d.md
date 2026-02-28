---
title: "Element Volume in 3D dfsu"
status: "Under Consideration"
category: "3D Mesh"
summary: "Compute element volumes for 3D layered meshes, enabling volume-weighted analysis."
---

## Value Proposition

Volume is the 3D equivalent of area — essential for computing total quantities (e.g., total pollutant mass in a water body), volume-weighted averages, and mass budgets. Currently only element areas are available, limiting 3D analysis capabilities.

## What This Enables

- **Volume-weighted averages**: Compute proper spatial averages in 3D domains
- **Mass budgets**: Calculate total mass of dissolved substances (concentration x volume)
- **Layer volumes**: Quantify the volume of individual layers or depth ranges
- **Temporal volume changes**: Track how volumes change with varying water levels

## Current State

Element areas are available in `GeometryFM3D` (inherited from 2D geometry). Layer thicknesses can be derived from node z-coordinates, but there is no method to compute element volumes directly.
