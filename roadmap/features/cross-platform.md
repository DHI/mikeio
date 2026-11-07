---
title: "Cross-Platform Support"
status: "Delivered"
category: "Infrastructure"
summary: "Native Python DFS I/O via mikecore, replacing the Windows-only pythonnet dependency."
---

## Value Proposition

MIKE IO was originally built on pythonnet, which required a .NET runtime and only worked on Windows. This blocked adoption by the many scientists and engineers who work on Linux or macOS, and made CI/CD pipelines fragile.

Migrating to mikecore — a Python module with bindings to DHI's C libraries — removed the .NET dependency entirely, enabling MIKE IO to run on any platform where Python runs.

## What This Enables

- **Linux and macOS support**: Use MIKE IO in any environment, including cloud compute and HPC clusters
- **Simpler installation**: No .NET runtime required — `pip install mikeio` just works
- **Reliable CI/CD**: Tests run on all platforms without special configuration
- **Docker-friendly**: Containerised workflows work out of the box

## Current Status

Delivered. All DFS file I/O uses mikecore. See ADR-002 for the decision record.
