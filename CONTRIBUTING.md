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
