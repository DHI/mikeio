---
title: "Time-Dependent Scaling"
status: "Under Consideration"
category: "Analysis"
summary: "Apply time-dependent adjustment factors (additive or multiplicative) to DFS file data, e.g. for climate change scenarios or tidal corrections."
---

## Value Proposition

Many workflows require adjusting time series or spatial fields by factors that vary over time — climate change impact assessments, tidal corrections, seasonal bias adjustment, or scenario scaling. A high-level API for applying time-dependent factors would eliminate repetitive boilerplate.

## What This Enables

- **Additive adjustments**: Add a time-varying delta to water levels, temperatures, or other variables
- **Multiplicative adjustments**: Scale rainfall or wind speed by time-dependent factors
- **Spatially varying factors**: Apply different factors to different regions or grid cells
- **Seasonal factors**: Apply month- or season-dependent adjustment factors
- **Scenario generation**: Produce multiple adjusted versions of a base dataset

## Current State

`scale()` and `transform()` in `generic.py` provide building blocks for applying mathematical operations to DFS file data. However, there is no high-level API for applying factors that vary by time step, season, or month.
