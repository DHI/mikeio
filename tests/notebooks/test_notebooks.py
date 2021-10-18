import os
import subprocess

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors import CellExecutionError

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.join(_TEST_DIR, "../..")
SKIP_LIST = []


def _process_notebook(notebook_filename, notebook_path="notebooks"):
    """Checks if an IPython notebook runs without error from start to finish. If so, writes the notebook to HTML (with outputs) and overwrites the .ipynb file (without outputs)."""
    with open(notebook_filename) as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")

    try:
        # Check that the notebook runs
        ep.preprocess(nb, {"metadata": {"path": notebook_path}})
    except CellExecutionError as e:
        print(f"Failed executing {notebook_filename}")
        print(e)
        raise

    print(f"Successfully executed {notebook_filename}")
    return


def _get_all_notebooks_in_repo(skip=[]):
    """Get all files .ipynb included in the git repository"""
    git_files = (
        subprocess.check_output(
            "git ls-tree --full-tree --name-only -r HEAD", shell=True
        )
        .decode("utf-8")
        .splitlines()
    )

    return [
        fn
        for fn in git_files
        if fn.endswith(".ipynb") and not any(s in fn for s in skip)
    ]


def test_notebook(notebook):
    _process_notebook(os.path.join(PARENT_DIR, notebook))


def pytest_generate_tests(metafunc):
    notebooks = _get_all_notebooks_in_repo(skip=SKIP_LIST)
    metafunc.parametrize("notebook", notebooks)
