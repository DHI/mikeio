# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MIKE IO is a Python package for reading, writing, and manipulating MIKE files (dfs0, dfs1, dfs2, dfs3, dfsu, mesh). It provides a high-level API for common data processing workflows with MIKE files from DHI.

## Development Commands

### Setup
```bash
# Install package in editable mode with dev dependencies
uv sync --group dev

# Or install all dependency groups (dev, test, notebooks)
uv sync --all-groups
```

### Testing
```bash
# Run all tests (excludes performance and notebook tests by default)
uv run pytest

# Run tests with coverage
uv run pytest --cov-report html --cov=mikeio tests/

# Run specific test file
uv run pytest tests/test_dfs0.py

# Run specific test function
uv run pytest tests/test_dfs0.py::test_repr

# Run performance tests
uv run pytest tests/performance/ --durations=0
```

### Code Quality
```bash
# Run all checks (lint, typecheck, test)
make check

# Lint code
uv run ruff check .

# Format code (required before committing)
uv run ruff format mikeio/

# Type checking
uv run mypy .
```

### Building
```bash
# Build package (includes typecheck and test)
make build
```

### Documentation
```bash
# Build and render documentation
make docs
```

## Code Architecture

### Source Structure
- **Source code location**: `src/mikeio/` (not `mikeio/` at root level)
- **Tests location**: `tests/`
- **Documentation**: `docs/`

### Core Modules

**`dfs/`** - DFS file format implementations
- `_dfs.py`: Base DFS functionality and common utilities
- `_dfs0.py`: Time series data (0D)
- `_dfs1.py`: 1D grid data
- `_dfs2.py`: 2D grid data
- `_dfs3.py`: 3D grid data

**`dfsu/`** - Flexible mesh (unstructured grid) implementations
- `_dfsu.py`: Base dfsu functionality
- `_layered.py`: Layered (3D) mesh files
- `_spectral.py`: Spectral wave data
- `_mesh.py`: Mesh-only files
- `_factory.py`: Factory for creating appropriate dfsu instances

**`dataset/`** - Core data structures
- `_dataset.py`: Dataset class - collection of DataArrays from a dfs file
- `_dataarray.py`: DataArray class - single item with data and metadata
- `_data_plot.py`: Plotting functionality for data structures

**`spatial/`** - Geometry and spatial operations
- `_grid_geometry.py`: Grid1D, Grid2D, Grid3D classes for structured grids
- `_FM_geometry.py`: GeometryFM2D for 2D flexible mesh
- `_FM_geometry_layered.py`: GeometryFM3D for 3D flexible mesh
- `_FM_geometry_spectral.py`: Geometry for spectral data
- `_geometry.py`: Base geometry classes
- `crs.py`: Coordinate reference system utilities

**`eum/`** - Engineering Unit Manager
- `_eum.py`: EUMType, EUMUnit, ItemInfo for physical quantities and units

**`pfs/`** - Parameter file system
- `_pfsdocument.py`: PfsDocument for reading MIKE parameter files
- `_pfssection.py`: PfsSection representing sections in parameter files

**Top-level modules**:
- `generic.py`: Generic functions for all dfs files (concat, extract, scale, etc.)
- `_time.py`: Time handling utilities
- `_track.py`: Track/transect functionality
- `_interpolation.py`: Interpolation utilities
- `_spectral.py`: Spectral analysis utilities

### Key Design Patterns

**Factory pattern**: Use `mikeio.open()` to get appropriate file type object (Dfs0, Dfs1, Dfsu2DH, etc.) based on file content.

**Read/write separation**: File objects (Dfs0, Dfsu, etc.) are lightweight metadata holders. Use `.read()` to load data into Dataset/DataArray objects.

**Composition over inheritance**: Dataset contains DataArrays; DataArray contains data, time, geometry, and item metadata. Geometry objects are composed into the data structures rather than inherited.

## Documentation Structure

The project uses **Quartodoc** and **Quarto** to build comprehensive documentation with user guides, examples, and API reference.

### Documentation Tools

- **Quartodoc** (v0.9.1): Generates API documentation from docstrings and type annotations
- **Quarto**: Static site generator that renders the final documentation website
- **Configuration**: `docs/_quarto.yml` defines the website structure, navigation, and quartodoc settings

### Building Documentation

```bash
# Build and render documentation (from project root)
make docs

# Manual build (from docs/ directory)
cd docs
uv run quartodoc build    # Generate API docs from source code
uv run quarto render      # Render the complete website
```

The build process:
1. `quartodoc build` reads the source code and generates API reference pages in `docs/api/`
2. `quarto render` processes all `.qmd` files (Quarto markdown) and builds the static site in `docs/_site/`

### Publishing Documentation

**IMPORTANT**: Documentation is published to GitHub Pages (https://dhi.github.io/mikeio/) **ONLY when a GitHub release is published**.

**Publishing workflow:**
1. Documentation is automatically deployed when you create a new release on GitHub
2. The `.github/workflows/docs.yml` workflow triggers on release publication
3. This ensures published docs always match the version available on PyPI

**Why publish on release only?**
- Prevents version mismatch: users installing from PyPI see docs for the version they installed
- Avoids documenting unreleased features that users can't access
- Keeps the published docs stable and predictable

**Testing documentation before release:**
```bash
# Build docs locally to verify examples work with current code
make docs

# View the rendered site
xdg-open docs/_site/index.html  # WSL2
# Or on Windows: start docs/_site/index.html
```

**Release process:**
1. Update version in `pyproject.toml` (remove `.dev0` suffix)
2. Commit and push changes
3. Create a new release on GitHub (this triggers both PyPI publish and docs deploy)
4. After release, bump version to next `.dev0` version

### Content Types

1. **User Guide** (`user-guide/*.qmd`): Conceptual documentation and tutorials
   - Written in Quarto markdown (`.qmd`)
   - Executable code blocks with outputs
   - Organized by file type and concept

2. **Examples** (`examples/*.qmd`): Real-world usage examples
   - Organized by file type (dfs0, dfs2, dfsu)
   - Each example is a complete, executable Quarto notebook
   - Uses symlinked test data from `tests/testdata/`

3. **API Reference** (`api/*.qmd`): Auto-generated from docstrings
   - Generated by `quartodoc build` - DO NOT edit manually
   - Re-generated on every docs build
   - Source docstrings are in `src/mikeio/` Python files

### Interlinks

The documentation supports cross-linking to external library documentation (numpy, xarray, pandas, scipy) using the `interlinks` filter configured in `_quarto.yml`.

## Data Flow Pattern

1. **Open**: `dfs = mikeio.open("file.dfs2")` → Returns Dfs2 object with metadata only
2. **Read**: `ds = dfs.read()` → Returns Dataset with all data loaded
3. **Select/Process**: Use Dataset/DataArray methods for subsetting and analysis
4. **Write**: `ds.to_dfs("output.dfs2")` → Write Dataset back to dfs file

Alternative: `ds = mikeio.read("file.dfs2")` combines open and read.

## Important Conventions

### Versioning
- **Development versions**: Use `.dev0` suffix (e.g., `3.1.0.dev0`) to indicate unreleased code
  - This clearly marks versions installed directly from GitHub as development/pre-release
  - Version is set in `pyproject.toml` line 8
- **Released versions**: Remove `.dev0` suffix for official releases (e.g., `3.0.0`, `3.1.0`)
- **Latest release**: v3.0.0 (December 2024)

### Dependencies
- Uses `mikecore` library (Python module with bindings to DFS and EUM C libraries) for low-level file I/O
- Do NOT use conda for installation (outdated version on conda-forge)
- Project uses **uv** for dependency management with `dependency-groups` in pyproject.toml
  - `dev`: Development tools (pytest, quartodoc, ruff, mypy, etc.)
  - `test`: Testing dependencies only
  - `notebooks`: Jupyter notebook dependencies

### Testing
- Uses pytest with type hints (all test functions must be typed)
- Test files in `tests/` mirror source structure
- Tests use fixtures from `conftest.py` and test data from `tests/testdata/`
- Type checking is enforced with mypy (excludes some test files, see pyproject.toml)

### Code Style
- Uses ruff for linting and formatting
- Docstrings required for public modules, classes, methods, functions (D100-D103)
- Line length limit is ignored (E501)
- Single letter variable names are allowed for common patterns (E741)

### Type Checking
- `disallow_untyped_defs = true` in mypy config
- All functions must have type annotations
- Return types must be explicit
- Some test files are excluded from mypy checks (see pyproject.toml)

## Testing Notes

When running individual tests, always use the full path from the repository root:
```bash
uv run pytest tests/test_dfs0.py::test_specific_function
```

Test data is located in `tests/testdata/` organized by file type.
- When creating a PR, start out in draft state