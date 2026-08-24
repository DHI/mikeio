---
title: "Consistent dfs2 and dfsu Plotting"
status: "Under Consideration"
category: "Visualization"
summary: "Unified plotting interface across grid and mesh geometries with consistent styling and options."
---

## Value Proposition

Currently, dfs2 and dfsu data use separate plotting code paths with different styling defaults and option names. A unified interface would make it easier to compare results across grid and mesh models, and reduce the learning curve for users working with both formats.

## What This Enables

- **Consistent API**: Same method signatures and options for grid and mesh plots
- **Easy comparison**: Plot dfs2 and dfsu results side by side with matching colour scales and styling
- **Reduced learning curve**: Learn one plotting interface instead of two
- **Shared defaults**: Consistent colormaps, label formatting, and layout across geometry types

## Current State

Plotting works well for both dfs2 and dfsu individually, but the implementations are separate (`_data_plot.py` for grids, `_FM_plot.py` for meshes) with different default styles and option handling.
