# This file contains tests involving several components and tests typical end-to-end workflows
# So not unit tests

from pathlib import Path
import mikeio


def test_read_file_multiply_two_items_and_save_to_new_file(tmp_path: Path) -> None:
    ds = mikeio.read("tests/testdata/oresundHD_run1.dfsu")

    da = (ds[0] * ds[1]).nanmax(axis="time")

    file_path = tmp_path / "mult.dfsu"

    da.to_dfs(file_path)


def test_aggregation_workflows(tmp_path: Path) -> None:
    dfs = mikeio.Dfsu2DH("tests/testdata/HD2D.dfsu")

    ds = dfs.read(items=["Surface elevation", "Current speed"])
    ds2 = ds.max(axis=1)

    outfilename = tmp_path / "max.dfs0"
    ds2.to_dfs(outfilename)
    assert outfilename.exists()

    ds3 = ds.min(axis=1)

    outfilename = tmp_path / "min.dfs0"
    ds3.to_dfs(outfilename)
    assert outfilename.exists()


def test_weighted_average(tmp_path: Path) -> None:
    dfs = mikeio.Dfsu2DH("tests/testdata/HD2D.dfsu")

    ds = dfs.read(items=["Surface elevation", "Current speed"])

    area = dfs.geometry.get_element_area()
    ds2 = ds.average(weights=area, axis=1)

    out_path = tmp_path / "average.dfs0"
    ds2.to_dfs(out_path)
    assert out_path.exists()
