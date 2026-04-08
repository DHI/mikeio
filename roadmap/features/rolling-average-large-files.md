---
title: "Out-of-Core Temporal Statistics"
status: "Under Consideration"
category: "Analysis"
summary: "Compute time-aggregated and rolling statistics on DFS files too large to fit in memory."
---

## Value Proposition

Long-duration or high-resolution simulations produce files that exceed available memory. Users need both full time-axis aggregation (mean, max, min, quantile over all timesteps) and rolling window statistics — all without loading the entire file.

## What This Enables

- **Full time-axis aggregation**: Compute mean, max, min, percentiles over the entire time series out-of-core
- **Rolling windows**: Rolling mean, min, max, and sum with configurable window sizes
- **Streaming output**: Write results to a new file as chunks are processed
- **Arbitrary file sizes**: Process multi-GB files with bounded memory usage

## Current State

`generic.py` already has chunked implementations of `avg_time` and `quantile` that process files in temporal chunks. These cover basic full-axis aggregation. Rolling window operations and a broader set of statistics (max, min, std) are not yet supported but could build on the same chunked infrastructure.
