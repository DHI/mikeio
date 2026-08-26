"""Tests that ``~`` is expanded in file names, like pandas and xarray do."""

import shutil
from pathlib import Path

import pytest

import mikeio
from mikeio import generic
from mikeio._path import normalize_path


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the user's home directory at a temporary directory."""
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.setenv("USERPROFILE", str(h))  # Windows
    assert Path("~").expanduser() == h
    return h


def copy_to_home(home: Path, testfile: str) -> str:
    """Copy a test data file into the fake home dir, return its ``~`` path."""
    src = Path("tests/testdata") / testfile
    shutil.copy(src, home / src.name)
    return f"~/{src.name}"


def test_normalize_path_expands_tilde(home: Path) -> None:
    assert normalize_path("~/wl.dfs0") == str(home / "wl.dfs0")
    assert normalize_path(Path("~/wl.dfs0")) == str(home / "wl.dfs0")


def test_normalize_path_leaves_other_paths_alone() -> None:
    assert normalize_path("tests/testdata/random.dfs0") == "tests/testdata/random.dfs0"
    assert normalize_path(Path("a/b.dfs0")) == str(Path("a/b.dfs0"))


def test_normalize_path_is_idempotent(home: Path) -> None:
    once = normalize_path("~/wl.dfs0")
    assert normalize_path(once) == once


@pytest.mark.parametrize(
    "testfile",
    ["random.dfs0", "tide1.dfs1", "gebco_sound.dfs2", "test_dfs3.dfs3", "HD2D.dfsu"],
)
def test_read_tilde(home: Path, testfile: str) -> None:
    ds = mikeio.read(copy_to_home(home, testfile))
    assert ds.n_items > 0


@pytest.mark.parametrize(
    "testfile",
    ["random.dfs0", "tide1.dfs1", "gebco_sound.dfs2", "test_dfs3.dfs3", "HD2D.dfsu"],
)
def test_open_tilde(home: Path, testfile: str) -> None:
    dfs = mikeio.open(copy_to_home(home, testfile))
    assert not isinstance(dfs, mikeio.Mesh)
    assert dfs.n_items > 0


@pytest.mark.parametrize(
    "testfile",
    ["random.dfs0", "tide1.dfs1", "gebco_sound.dfs2", "test_dfs3.dfs3", "HD2D.dfsu"],
)
def test_to_dfs_tilde(home: Path, testfile: str) -> None:
    ds = mikeio.read(Path("tests/testdata") / testfile)
    outfilename = f"~/out{Path(testfile).suffix}"

    ds.to_dfs(outfilename)

    assert (home / Path(outfilename).name).exists()
    assert not Path("~").exists()  # no literal '~' directory in the cwd
    assert mikeio.read(outfilename).n_items == ds.n_items


def test_dataarray_to_dfs_tilde(home: Path) -> None:
    da = mikeio.read("tests/testdata/random.dfs0")[0]

    da.to_dfs("~/out.dfs0")

    assert (home / "out.dfs0").exists()
    assert not Path("~").exists()


def test_open_mesh_tilde(home: Path) -> None:
    msh = mikeio.open(copy_to_home(home, "odense_rough.mesh"))
    assert isinstance(msh, mikeio.Mesh)
    assert msh.n_elements > 0


def test_mesh_write_tilde(home: Path) -> None:
    msh = mikeio.Mesh("tests/testdata/odense_rough.mesh")

    msh.write("~/out.mesh")

    assert (home / "out.mesh").exists()
    assert not Path("~").exists()
    assert mikeio.Mesh("~/out.mesh").n_elements == msh.n_elements


def test_geometry_to_mesh_tilde(home: Path) -> None:
    ds = mikeio.read("tests/testdata/HD2D.dfsu")

    ds.geometry.to_mesh("~/out.mesh")

    assert (home / "out.mesh").exists()
    assert not Path("~").exists()


def test_grid2d_to_mesh_tilde(home: Path) -> None:
    ds = mikeio.read("tests/testdata/gebco_sound.dfs2")

    ds.geometry.to_mesh("~/out.mesh")

    assert (home / "out.mesh").exists()
    assert not Path("~").exists()


def test_read_pfs_tilde(home: Path) -> None:
    src = Path("tests/testdata/pfs/lake.sw")
    shutil.copy(src, home / src.name)

    pfs = mikeio.read_pfs("~/lake.sw")

    assert pfs.targets is not None


def test_pfs_write_tilde(home: Path) -> None:
    pfs = mikeio.read_pfs("tests/testdata/pfs/lake.sw")

    pfs.write("~/out.sw")

    assert (home / "out.sw").exists()
    assert not Path("~").exists()


def test_generic_scale_tilde(home: Path) -> None:
    infilename = copy_to_home(home, "random.dfs0")

    generic.scale(infilename, "~/out.dfs0", factor=2.0)

    assert (home / "out.dfs0").exists()
    assert not Path("~").exists()


def test_generic_extract_tilde(home: Path) -> None:
    infilename = copy_to_home(home, "random.dfs0")

    generic.extract(infilename, "~/out.dfs0")

    assert (home / "out.dfs0").exists()
    assert not Path("~").exists()


def test_generic_concat_tilde(home: Path) -> None:
    infilename = copy_to_home(home, "tide1.dfs1")

    generic.concat([infilename, infilename], "~/out.dfs1")

    assert (home / "out.dfs1").exists()
    assert not Path("~").exists()


def test_generic_diff_tilde(home: Path) -> None:
    infilename = copy_to_home(home, "tide1.dfs1")

    generic.diff(infilename, infilename, "~/out.dfs1")

    assert (home / "out.dfs1").exists()
    assert not Path("~").exists()


def test_generic_avg_time_tilde(home: Path) -> None:
    infilename = copy_to_home(home, "tide1.dfs1")

    generic.avg_time(infilename, "~/out.dfs1")

    assert (home / "out.dfs1").exists()
    assert not Path("~").exists()


def test_generic_quantile_tilde(home: Path) -> None:
    infilename = copy_to_home(home, "tide1.dfs1")

    generic.quantile(infilename, "~/out.dfs1", q=0.5)

    assert (home / "out.dfs1").exists()
    assert not Path("~").exists()


def test_generic_change_datatype_tilde(home: Path) -> None:
    infilename = copy_to_home(home, "HD2D.dfsu")

    generic.change_datatype(infilename, "~/out.dfsu", datatype=107)

    assert (home / "out.dfsu").exists()
    assert not Path("~").exists()


def test_generic_fill_corrupt_tilde(home: Path) -> None:
    infilename = copy_to_home(home, "tide1.dfs1")

    generic.fill_corrupt(infilename, "~/out.dfs1")

    assert (home / "out.dfs1").exists()
    assert not Path("~").exists()


def test_extract_track_tilde(home: Path) -> None:
    src = Path("tests/testdata/altimetry_NorthSea_20171027.csv")
    shutil.copy(src, home / src.name)
    dfs = mikeio.Dfsu2DH("tests/testdata/NorthSea_HD_and_windspeed.dfsu")

    ds = dfs.extract_track(f"~/{src.name}")

    assert ds.n_items > 0


def test_missing_file_error_shows_expanded_path(home: Path) -> None:
    with pytest.raises(FileNotFoundError, match=str(home)):
        mikeio.open("~/does_not_exist.dfs0")
