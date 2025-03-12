from pathlib import Path


import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors import CellExecutionError


def _process_notebook(fp: Path):
    """Checks if an IPython notebook runs without error from start to finish. If so, writes the notebook to HTML (with outputs) and overwrites the .ipynb file (without outputs)."""
    with open(fp) as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")

    try:
        # Check that the notebook runs
        ep.preprocess(nb, {"metadata": {"path": "notebooks"}})
    except CellExecutionError as e:
        print(f"Failed executing {fp}")
        print(e)
        raise

    print(f"Successfully executed {fp}")
    return


def _get_all_notebooks_in_repo() -> list[Path]:
    ROOT_DIR = Path(__file__).parent.parent.parent
    NOTEBOOK_DIR = ROOT_DIR / "notebooks"

    return list(NOTEBOOK_DIR.glob("*.ipynb"))


def test_notebook(notebook):
    _process_notebook(notebook)


def pytest_generate_tests(metafunc):
    notebooks = _get_all_notebooks_in_repo()
    metafunc.parametrize("notebook", notebooks)


if __name__ == "__main__":
    notebooks = _get_all_notebooks_in_repo()
    for notebook in notebooks:
        print(notebook)
