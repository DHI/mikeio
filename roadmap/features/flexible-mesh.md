---
title: "Flexible Mesh (dfsu) Support"
status: "Delivered"
category: "File Formats"
summary: "Full read/write support for 2D and 3D unstructured mesh files, including spectral data."
---

## Value Proposition

Flexible mesh (dfsu) files are the primary output format for MIKE 21/3 FM simulations. Supporting these files — including their complex 3D layered structure and spectral variants — is essential for post-processing simulation results in Python.

## What This Enables

- **Read and write** 2D and 3D dfsu files with element-centred or node-centred data
- **Spatial subsetting**: Extract data by element index, coordinate, or bounding box
- **Geometry access**: Node coordinates, element tables, boundary codes, projection info
- **3D layered meshes**: Navigate vertical structure with layer-aware operations
- **Spectral data**: Read directional and frequency spectra from spectral dfsu files
- **Mesh manipulation**: Create, modify, and write mesh files

## Current Status

Delivered. Full factory-based architecture supports all dfsu data types (2001, 2002, 2003, 2004) with geometry-aware operations.
