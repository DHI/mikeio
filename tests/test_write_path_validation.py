"""Tests that writing to a path that cannot be created raises a clear error.

The underlying mikecore library does not check the output path: writing to a
path it cannot create used to crash the interpreter.
"""

from pathlib import Path

import pytest

import mikeio
from mikeio import generic

FILES_BY_EXTENSION = {
    ".dfs0": "da_diagnostic.dfs0",
    ".dfs1": "tide1.dfs1",
    ".dfs2": "gebco_sound.dfs2",
    ".dfs3": "test_dfs3.dfs3",
    ".dfsu": "oresundHD_run1.dfsu",
}


@pytest.mark.parametrize("extension", list(FILES_BY_EXTENSION))
def test_to_dfs_creates_missing_directory(tmp_path: Path, extension: str) -> None:
    ds = mikeio.read("tests/testdata/" + FILES_BY_EXTENSION[extension])
    outfilename = tmp_path / "new_folder" / f"out{extension}"

    ds.to_dfs(outfilename)

    assert outfilename.exists()


@pytest.mark.parametrize("extension", list(FILES_BY_EXTENSION))
def test_to_dfs_impossible_path(tmp_path: Path, extension: str) -> None:
    ds = mikeio.read("tests/testdata/" + FILES_BY_EXTENSION[extension])
    not_a_folder = tmp_path / "a_file"
    not_a_folder.touch()

    with pytest.raises(OSError):
        ds.to_dfs(not_a_folder / f"out{extension}")


def test_to_dfs_existing_directory(tmp_path: Path) -> None:
    ds = mikeio.read("tests/testdata/gebco_sound.dfs2")
    folder = tmp_path / "out.dfs2"
    folder.mkdir()

    with pytest.raises(IsADirectoryError):
        ds.to_dfs(folder)


def test_mesh_write_impossible_path(tmp_path: Path) -> None:
    msh = mikeio.Mesh("tests/testdata/odense_rough.mesh")
    not_a_folder = tmp_path / "a_file"
    not_a_folder.touch()

    with pytest.raises(OSError):
        msh.write(not_a_folder / "out.mesh")


def test_generic_scale_impossible_path(tmp_path: Path) -> None:
    not_a_folder = tmp_path / "a_file"
    not_a_folder.touch()

    with pytest.raises(OSError):
        generic.scale(
            "tests/testdata/gebco_sound.dfs2",
            not_a_folder / "out.dfs2",
            factor=2.0,
        )
