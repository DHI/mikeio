# Guidelines for contribution

1. Clone the repo
2. Install package in editable mode with development dependencies: `uv sync --group dev` (or `uv sync --all-groups` for all dependency groups)
3. If you are fixing a bug, first add a failing test
4. Make changes
5. Verify that all tests passes by running `uv run pytest` from the package root directory
6. Format the code by running `uv run ruff format .`
7. Make a pull request with a summary of the changes
