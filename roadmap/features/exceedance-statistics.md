---
title: "Exceedance Statistics"
status: "Under Consideration"
category: "Analysis"
summary: "Compute exceedance probabilities, return periods, and threshold-based statistics directly on DataArrays."
---

## Value Proposition

Exceedance statistics — how often a value exceeds a threshold, or what value is exceeded X% of the time — are fundamental to coastal and hydraulic engineering design. Providing these as first-class operations on DataArray would eliminate boilerplate and reduce errors in common workflows.

## What This Enables

- **Exceedance curves**: Compute and plot percentage-of-time-exceeded for any variable
- **Return periods**: Estimate N-year return values from time series
- **Threshold statistics**: Count exceedances, compute durations above/below thresholds
- **Spatial maps**: Produce maps of exceedance values across a domain (e.g., "100-year water level")

## Current State

`quantile()` is available on DataArray for computing percentiles, which is a building block for exceedance statistics. However, there is no dedicated exceedance API with return-period estimation or threshold-duration analysis.
