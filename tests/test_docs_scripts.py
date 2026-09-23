import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np

import mikeio

SCRIPTS = Path(__file__).parent.parent / "docs" / "scripts"
TESTDATA = Path(__file__).parent / "testdata"


def load_script(name: str) -> ModuleType:
    """Import a docs script as a module, running against the installed mikeio, not its pinned one."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_diff_subtracts_baseline_from_scenario(tmp_path: Path) -> None:
    scenario = TESTDATA / "oresundHD_run2.dfsu"
    baseline = TESTDATA / "oresundHD_run1.dfsu"
    out = tmp_path / "diff.dfsu"
    load_script("diff").main([str(scenario), str(baseline), str(out)])

    expected = mikeio.read(scenario)[0].to_numpy() - mikeio.read(baseline)[0].to_numpy()
    np.testing.assert_allclose(mikeio.read(out)[0].to_numpy(), expected)
