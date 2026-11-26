import subprocess
import sys
from pathlib import Path

TESTDATA = Path(__file__).parent / "testdata"


def test_reading_files_does_not_import_matplotlib_or_scipy() -> None:
    # sys.modules of the test process already holds both, hence a new interpreter
    code = f"""
import sys
import mikeio

for name in ("random.dfs0", "eq.dfs2", "HD2D.dfsu"):
    mikeio.read({str(TESTDATA)!r} + "/" + name)

print(",".join(m for m in ("matplotlib", "scipy") if m in sys.modules))
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    loaded = result.stdout.strip()
    assert loaded == "", (
        f"import mikeio + read() imported {loaded}; "
        "import it inside the function that uses it (see adr/014-lazy-imports.md)"
    )
