# This file contains tests involving several components and tests typical end-to-end workflows
# So not unit tests

from pathlib import Path
import mikeio


def test_read_file_multiply_two_items_and_save_to_new_file(tmp_path: Path) -> None:
    ds = mikeio.read("tests/testdata/oresundHD_run1.dfsu")

    da = (ds[0] * ds[1]).nanmax(axis="time")

    file_path = tmp_path / "mult.dfsu"

    da.to_dfs(file_path)
