import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

import mikeio

SCRIPTS = Path(__file__).parent.parent / "docs" / "scripts"


def load_script(name: str) -> ModuleType:
    """Import a docs script as a module, running against the installed mikeio, not its pinned one."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_concat_expands_glob(tmp_path: Path) -> None:
    out = tmp_path / "out.dfs1"
    load_script("concat").main(["tests/testdata/tide[12].dfs1", str(out)])

    ds = mikeio.read(out)
    t1 = mikeio.read("tests/testdata/tide1.dfs1").time
    t2 = mikeio.read("tests/testdata/tide2.dfs1").time
    assert ds.time[0] == t1[0]
    assert ds.time[-1] == t2[-1]


def test_concat_exits_nonzero_when_nothing_matches(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as e:
        load_script("concat").main(["no_such_*.dfs1", str(tmp_path / "out.dfs1")])
    assert e.value.code != 0
