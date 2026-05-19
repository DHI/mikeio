---
title: "PFS File Support"
status: "Delivered"
category: "File Formats"
summary: "Read, modify, and write MIKE parameter files (.pfs) as structured Python objects."
---

## Value Proposition

MIKE parameter files (.pfs) control simulation setup — boundary conditions, output specifications, solver parameters. Being able to read and modify these files programmatically enables batch simulation workflows, parameter studies, and automated model setup.

## What This Enables

- **Read PFS files** into a structured PfsDocument with named sections and key-value pairs
- **Modify parameters** using familiar Python dict-like access
- **Write PFS files** back to disk with correct formatting
- **Automate model setup**: Script parameter sweeps, boundary condition updates, and configuration changes
- **Extract metadata**: Read output specifications, timestep settings, and other configuration from existing setups

## Current Status

Delivered. PfsDocument and PfsSection provide full read/write support for the PFS format.
