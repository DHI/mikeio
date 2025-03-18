import pytest
import subprocess
import pathlib

MARIMO_DIR = pathlib.Path("marimo")
notebooks = list(MARIMO_DIR.glob("*.py"))


@pytest.mark.parametrize("notebook", notebooks, ids=[str(s.name) for s in notebooks])
def test_marimo_execution(notebook):
    """Test that each Python script runs without errors, setting the correct working directory."""
    result = subprocess.run(
        ["python", notebook.name],
        cwd=MARIMO_DIR,
        capture_output=True,
        text=True,
    )
    assert (
        result.returncode == 0
    ), f"Marimo notebook: {notebook.name} failed with error:\n{result.stderr}"
