# Guidelines for contribution

1. Clone the repo
2. Install [uv](https://docs.astral.sh/uv/) and [just](https://github.com/casey/just) if you don't have them already
3. Install package in editable mode with development dependencies: `uv sync --group dev` (or `uv sync --all-groups` for all dependency groups)
4. If you are fixing a bug, first add a failing test
5. Make changes
6. Verify that all tests pass by running `just test` from the package root directory
7. Format the code by running `just format`
8. Make a pull request with a summary of the changes

Run `just --list` to see all available recipes.

## Supported Python versions

MIKE IO follows [SPEC 0](https://scientific-python.org/specs/spec-0000/), the Scientific
Python ecosystem's support policy (which superseded [NEP 29](https://numpy.org/neps/nep-0029-deprecation_policy.html)):

- **Python versions** are supported for **36 months** after their release, then dropped.
- **Core dependencies** (NumPy, pandas, SciPy, …) are supported for **24 months** after release.

In practice this means the minimum supported Python tracks the SPEC 0 schedule rather than
the latest end-of-life date. When raising the floor, update `requires-python`, the
`python_version` under `[tool.mypy]`, the Python classifiers in `pyproject.toml`, and the
CI matrices in `.github/workflows/`. Dropping a Python version is a routine, scheduled
change — not a breaking feature removal — and it lets us adopt dependency releases that
require a newer interpreter.
