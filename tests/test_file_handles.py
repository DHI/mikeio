"""No native file handle outlives the call that opened it.

Every test counts the descriptors this process still holds on the files it
touched (see the `count_fds` fixture in conftest.py) after an open, a read, an
invalid argument, an exception forced part-way, a write or a generic function.
The file objects returned by mikeio.open() are kept alive while counting: they
are metadata holders and must not keep a handle for their lifetime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import pytest
from mikecore.DfsFile import DfsFile

import mikeio
from mikeio import generic
from mikeio.exceptions import ItemsError
from mikeio.generic import DerivedItem

CountFds = Callable[[str | Path], int]

TESTDATA = Path("tests/testdata")

DFS_FILES = {
    "dfs0": TESTDATA / "random.dfs0",
    "dfs1": TESTDATA / "random.dfs1",
    "dfs2": TESTDATA / "random.dfs2",
    "dfs3": TESTDATA / "test_dfs3.dfs3",
    "dfsu2d": TESTDATA / "HD2D.dfsu",
    "dfsu3d": TESTDATA / "basin_3d.dfsu",
    "dfsu2dv": TESTDATA / "oresund_vertical_slice.dfsu",
    "spectral": TESTDATA / "spectra" / "area_spectra.dfsu",
}
FILES = {**DFS_FILES, "mesh": TESTDATA / "odense_rough.mesh"}


class _Boom(RuntimeError):
    """Raised by a patched mikecore method to interrupt a call part-way."""


def _raise_on_call(
    monkeypatch: pytest.MonkeyPatch, cls: type, name: str, n: int
) -> None:
    """Patch cls.name to raise on its n-th call (1-based)."""
    original = getattr(cls, name)
    calls = 0

    def patched(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == n:
            raise _Boom(f"{name} call {n}")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(cls, name, patched)


@pytest.fixture
def read_raises(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Make the second timestep read of any dfs file fail."""
    _raise_on_call(monkeypatch, DfsFile, "ReadItemTimeStep", 2)
    # dfs0 reads all data in one call
    _raise_on_call(monkeypatch, DfsFile, "ReadDfs0DataDouble", 1)
    yield


# ---------------------------------------------------------------------------
# File objects and their readers


@pytest.mark.parametrize("kind", FILES)
def test_open_keeps_no_handle(kind: str, count_fds: CountFds) -> None:
    fn = FILES[kind]
    dfs: Any = mikeio.open(fn)
    assert count_fds(fn) == 0
    del dfs


@pytest.mark.parametrize("kind", DFS_FILES)
def test_read_keeps_no_handle(kind: str, count_fds: CountFds) -> None:
    fn = DFS_FILES[kind]
    dfs: Any = mikeio.open(fn)
    dfs.read()
    assert count_fds(fn) == 0


@pytest.mark.parametrize("kind", DFS_FILES)
def test_read_invalid_items_keeps_no_handle(kind: str, count_fds: CountFds) -> None:
    fn = DFS_FILES[kind]
    dfs: Any = mikeio.open(fn)
    with pytest.raises(ItemsError) as excinfo:
        dfs.read(items=99)
    # excinfo holds the traceback, and so every frame the error passed through
    assert excinfo.value is not None
    assert count_fds(fn) == 0


@pytest.mark.parametrize("kind", DFS_FILES)
def test_read_invalid_time_keeps_no_handle(kind: str, count_fds: CountFds) -> None:
    fn = DFS_FILES[kind]
    dfs: Any = mikeio.open(fn)
    with pytest.raises((IndexError, ValueError, KeyError)) as excinfo:
        dfs.read(time=10_000)
    assert excinfo.value is not None
    assert count_fds(fn) == 0


@pytest.mark.parametrize("kind", DFS_FILES)
def test_read_interrupted_keeps_no_handle(
    kind: str, count_fds: CountFds, read_raises: None
) -> None:
    fn = DFS_FILES[kind]
    dfs: Any = mikeio.open(fn)
    with pytest.raises(_Boom) as excinfo:
        dfs.read()
    assert excinfo.value is not None
    assert count_fds(fn) == 0


def test_dfs0_derived_properties_keep_no_handle(count_fds: CountFds) -> None:
    fn = TESTDATA / "random.dfs0"
    dfs = mikeio.Dfs0(fn)
    assert dfs.end_time is not None
    assert dfs.timestep > 0
    assert len(dfs.time) == dfs.n_timesteps
    assert count_fds(fn) == 0


def test_dfs123_timestep_keeps_no_handle(count_fds: CountFds) -> None:
    fn = TESTDATA / "random.dfs2"
    dfs = mikeio.Dfs2(fn)
    assert dfs.timestep > 0
    assert dfs.end_time is not None
    assert count_fds(fn) == 0


def _truncated_copy(src: Path, tmp_path: Path) -> Path:
    fp = tmp_path / f"corrupt_{src.name}"
    fp.write_bytes(src.read_bytes()[:-200])
    return fp


def test_dfsu_read_corrupt_raises_keeps_no_handle(
    tmp_path: Path, count_fds: CountFds
) -> None:
    fp = _truncated_copy(TESTDATA / "HD2D.dfsu", tmp_path)
    dfs: Any = mikeio.open(fp)
    with pytest.raises(ValueError, match="timestep 8") as excinfo:
        dfs.read()
    assert excinfo.value is not None
    assert count_fds(fp) == 0


def test_dfsu_read_corrupt_filled_keeps_no_handle(
    tmp_path: Path, count_fds: CountFds
) -> None:
    """The reader reopens the file after a corrupt timestep; that one is closed too."""
    fp = _truncated_copy(TESTDATA / "HD2D.dfsu", tmp_path)
    dfs: Any = mikeio.open(fp)
    with pytest.warns(UserWarning, match="timestep 8"):
        ds = dfs.read(error_bad_data=False, fill_bad_data_value=-1.0)
    assert (ds[-1].values[-1] == -1.0).all()
    assert count_fds(fp) == 0


def test_dfsu_extract_track_keeps_no_handle(count_fds: CountFds) -> None:
    fn = TESTDATA / "NorthSea_HD_and_windspeed.dfsu"
    dfs: Any = mikeio.open(fn)
    dfs.extract_track(TESTDATA / "altimetry_NorthSea_20171027.csv")
    assert count_fds(fn) == 0


# ---------------------------------------------------------------------------
# Writers and append


@pytest.mark.parametrize("kind", ["dfs0", "dfs1", "dfs2", "dfs3", "dfsu2d", "dfsu3d"])
def test_write_keeps_no_handle(kind: str, tmp_path: Path, count_fds: CountFds) -> None:
    fn = DFS_FILES[kind]
    out = tmp_path / f"out{fn.suffix}"
    mikeio.read(fn).to_dfs(out)
    assert count_fds(out) == 0


@pytest.mark.parametrize("kind", ["dfs0", "dfs1", "dfs2", "dfs3", "dfsu2d", "dfsu3d"])
def test_write_interrupted_keeps_no_handle(
    kind: str, tmp_path: Path, count_fds: CountFds, monkeypatch: pytest.MonkeyPatch
) -> None:
    fn = DFS_FILES[kind]
    ds = mikeio.read(fn)
    out = tmp_path / f"out{fn.suffix}"
    _raise_on_call(monkeypatch, DfsFile, "WriteItemTimeStepNext", 2)
    _raise_on_call(monkeypatch, DfsFile, "WriteDfs0DataDouble", 1)
    with pytest.raises(_Boom) as excinfo:
        ds.to_dfs(out)
    assert excinfo.value is not None
    assert count_fds(out) == 0


@pytest.mark.parametrize(
    "fn",
    [
        TESTDATA / "eq.dfs2",
        TESTDATA / "Karup_MIKE_SHE_head_output.dfs3",
        TESTDATA / "consistency" / "oresundHD.dfsu",
        TESTDATA / "oresund_sigma_z.dfsu",
    ],
    ids=lambda p: p.name,
)
def test_append_keeps_no_handle(fn: Path, tmp_path: Path, count_fds: CountFds) -> None:
    ds = mikeio.read(fn, time=[0])
    ds2 = mikeio.read(fn, time=[1])
    out = tmp_path / f"appended{fn.suffix}"
    ds.to_dfs(out)
    dfs: Any = mikeio.open(out)
    dfs.append(ds2)
    assert count_fds(out) == 0
    # the file object follows the appended time axis, as when it reopened the file
    assert dfs.n_timesteps == 2
    assert dfs.read().n_timesteps == 2
    assert count_fds(out) == 0


# ---------------------------------------------------------------------------
# generic


def _files(*names: str) -> list[Path]:
    return [TESTDATA / n for n in names]


GENERIC_CALLS: dict[str, tuple[list[Path], Callable[[list[Path], Path], None]]] = {
    "concat_last": (
        _files("tide1.dfs1", "tide2_offset.dfs1", "tide4.dfs1"),
        lambda i, o: generic.concat(i, o, keep="last"),
    ),
    "concat_first": (
        _files("tide1.dfs1", "tide2_offset.dfs1", "tide4.dfs1"),
        lambda i, o: generic.concat(i, o, keep="first"),
    ),
    "concat_average": (
        _files("tide1.dfs1", "tide2_offset.dfs1", "tide4.dfs1"),
        lambda i, o: generic.concat(i, o, keep="average"),
    ),
    "extract": (
        _files("waves.dfs2"),
        lambda i, o: generic.extract(i[0], o, start=1, end=-1),
    ),
    "extract_items_dfsu3d": (
        _files("oresund_vertical_slice.dfsu"),
        lambda i, o: generic.extract(i[0], o, items="Temperature"),
    ),
    "extract_stops_early": (
        _files("da_diagnostic.dfs0"),
        lambda i, o: generic.extract(
            i[0], o, start="2017-10-27 01:58", end="2017-10-27 04:32"
        ),
    ),
    "avg_time": (_files("waves.dfs2"), lambda i, o: generic.avg_time(i[0], o)),
    "quantile": (
        _files("oresundHD_run1.dfsu"),
        lambda i, o: generic.quantile(i[0], o, q=[0.1, 0.9]),
    ),
    "quantile_dfsu3d": (
        _files("oresund_sigma_z.dfsu"),
        lambda i, o: generic.quantile(i[0], o, q=0.5),
    ),
    "diff": (
        _files("oresundHD_run1.dfsu", "oresundHD_run2.dfsu"),
        lambda i, o: generic.diff(i[0], i[1], o),
    ),
    "add": (
        _files("oresundHD_run1.dfsu", "oresundHD_run2.dfsu"),
        lambda i, o: generic.add(i[0], i[1], o),
    ),
    "scale": (
        _files("waves.dfs2"),
        lambda i, o: generic.scale(i[0], o, offset=1.0, factor=2.0),
    ),
    "fill_corrupt": (
        _files("waves.dfs2"),
        lambda i, o: generic.fill_corrupt(i[0], o, items=[1]),
    ),
    "change_datatype": (
        _files("oresundHD_run1.dfsu"),
        lambda i, o: generic.change_datatype(i[0], o, datatype=107),
    ),
    "transform": (
        _files("oresundHD_run1.dfsu"),
        lambda i, o: generic.transform(
            i[0],
            o,
            [
                DerivedItem(
                    name="Current Speed",
                    type=mikeio.EUMType.Current_Speed,
                    func=lambda x: np.sqrt(x["U velocity"] ** 2 + x["V velocity"] ** 2),
                )
            ],
        ),
    ),
}


def _assert_no_handles(count_fds: CountFds, files: list[Path]) -> None:
    for f in files:
        if f.exists():
            assert count_fds(f) == 0, f


@pytest.mark.parametrize("name", GENERIC_CALLS)
def test_generic_keeps_no_handle(
    name: str, tmp_path: Path, count_fds: CountFds
) -> None:
    infiles, call = GENERIC_CALLS[name]
    out = tmp_path / f"out{infiles[0].suffix}"
    call(infiles, out)
    _assert_no_handles(count_fds, [*infiles, out])


@pytest.mark.parametrize("name", GENERIC_CALLS)
def test_generic_interrupted_keeps_no_handle(
    name: str,
    tmp_path: Path,
    count_fds: CountFds,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    infiles, call = GENERIC_CALLS[name]
    out = tmp_path / f"out{infiles[0].suffix}"
    _raise_on_call(monkeypatch, DfsFile, "ReadItemTimeStep", 2)
    _raise_on_call(monkeypatch, DfsFile, "ReadItemTimeStepNext", 2)
    with pytest.raises(_Boom) as excinfo:
        call(infiles, out)
    assert excinfo.value is not None
    _assert_no_handles(count_fds, [*infiles, out])


@pytest.mark.parametrize(
    "call",
    [
        lambda i, o: generic.extract(i, o, items=[99]),
        lambda i, o: generic.scale(i, o, items=[99]),
        lambda i, o: generic.fill_corrupt(i, o, items=[99]),
        lambda i, o: generic.quantile(i, o, q=0.5, items=[99]),
    ],
    ids=["extract", "scale", "fill_corrupt", "quantile"],
)
def test_generic_invalid_items_keeps_no_handle(
    call: Callable[[Path, Path], None], tmp_path: Path, count_fds: CountFds
) -> None:
    infile = TESTDATA / "waves.dfs2"
    out = tmp_path / "out.dfs2"
    with pytest.raises(ItemsError) as excinfo:
        call(infile, out)
    assert excinfo.value is not None
    _assert_no_handles(count_fds, [infile, out])


def test_generic_extract_invalid_time_keeps_no_handle(
    tmp_path: Path, count_fds: CountFds
) -> None:
    infile = TESTDATA / "eq_relative.dfs0"
    out = tmp_path / "out.dfs0"
    # a relative time axis is not supported; raised as AttributeError today
    with pytest.raises(Exception) as excinfo:
        generic.extract(infile, out, start=0, end=4)
    assert excinfo.value is not None
    _assert_no_handles(count_fds, [infile, out])


def test_generic_concat_gap_keeps_no_handle(
    tmp_path: Path, count_fds: CountFds
) -> None:
    infiles = _files("tide1.dfs1", "tide4.dfs1")
    out = tmp_path / "concat.dfs1"
    with pytest.raises(Exception, match="Gap in time axis") as excinfo:
        generic.concat(infiles, out)
    assert excinfo.value is not None
    assert not out.exists()
    _assert_no_handles(count_fds, infiles)


def test_generic_transform_missing_item_keeps_no_handle(
    tmp_path: Path, count_fds: CountFds
) -> None:
    infile = TESTDATA / "oresundHD_run1.dfsu"
    out = tmp_path / "out.dfsu"
    items = [DerivedItem(name="Not relevant", func=lambda x: x["not in the file"])]
    with pytest.raises(KeyError, match="U velocity") as excinfo:
        generic.transform(infile, out, items)
    assert excinfo.value is not None
    assert not out.exists()
    _assert_no_handles(count_fds, [infile])
