"""Custom blocks of a dfs file header.

A dfs header block belongs to the file rather than to an item or a geometry, and
the same code writes it for dfs0, dfs1, dfs2 and dfs3, so it is tested here once
instead of per format. dfs2 supplies most of the test data: it is the format
whose "M21_Misc" block users actually edit, and tests/testdata has both a file
that carries one and a file that carries none.

tests/testdata inventory used below:
  BW_Ronne_Layout1998_rotated.dfs2  M21_Misc float32[7], land value 5.0
  gebco_sound.dfs2                  no custom blocks

Dataset-level semantics (ownership, propagation through operations) live in
test_dataset.py; dfs2 round-trip fidelity also rides
test_dfs2.py::is_header_unchanged_on_read_write.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import mikeio

M21_MISC = "M21_Misc"
ROTATED = "tests/testdata/BW_Ronne_Layout1998_rotated.dfs2"
NO_BLOCKS = "tests/testdata/gebco_sound.dfs2"

# One file per format that carries a custom block, so the shared read/write code
# is covered for all of them rather than for dfs2 alone.
BLOCK_FILES = [
    pytest.param(mikeio.Dfs0, "tests/testdata/sw_points.dfs0", id="dfs0"),
    pytest.param(mikeio.Dfs1, "tests/testdata/tide1.dfs1", id="dfs1"),
    pytest.param(mikeio.Dfs2, ROTATED, id="dfs2"),
    pytest.param(mikeio.Dfs3, "tests/testdata/Grid1.dfs3", id="dfs3"),
]


# === Reading ===


@pytest.mark.parametrize("cls,path", BLOCK_FILES)
def test_custom_blocks_roundtrip(cls: Any, path: str, tmp_path: Path) -> None:
    """Every dfs format reads its blocks and writes them back unchanged.

    Values must be bit-identical, not merely close: they are opaque metadata that
    MIKE tools read literally.
    """
    ds = mikeio.read(path)
    assert list(ds.custom_blocks) == [M21_MISC]
    assert cls(path).custom_blocks.keys() == ds.custom_blocks.keys()

    ds.custom_blocks["Mine"] = np.array([1.0], dtype=np.float32)

    fp = tmp_path / ("with_blocks" + Path(path).suffix)
    ds.to_dfs(fp)

    back = cls(fp).custom_blocks
    assert back.keys() == ds.custom_blocks.keys()
    for name, values in ds.custom_blocks.items():
        np.testing.assert_array_equal(back[name], values)
        assert back[name].dtype == values.dtype


def test_read_custom_blocks_from_file_object() -> None:
    dfs = mikeio.Dfs2(ROTATED)

    assert list(dfs.custom_blocks) == [M21_MISC]
    block = dfs.custom_blocks[M21_MISC]
    assert block.dtype == np.float32
    assert block.size == 7
    assert block[0] == pytest.approx(-22.5)  # orientation
    assert block[2] == pytest.approx(-900.0)  # geographic flag
    assert block[3] == pytest.approx(5.0)  # land value


@pytest.mark.parametrize("cls,path", BLOCK_FILES)
def test_file_object_custom_blocks_are_read_only(cls: Any, path: str) -> None:
    """A file object cannot write a header, so an edit through one must raise.

    Handing out plain copies would let the edit look like it worked, so both the
    mapping and its arrays refuse it.
    """
    dfs = cls(path)
    blocks = dfs.custom_blocks
    name = next(iter(blocks))

    with pytest.raises(TypeError):
        blocks["New"] = np.array([1.0], dtype=np.float32)
    with pytest.raises(ValueError, match="read-only"):
        blocks[name][0] = 0.0

    # the values are copies as well, so even unfreezing one cannot reach the file
    original = blocks[name][0]
    escaped = dfs.custom_blocks[name]
    escaped.setflags(write=True)
    escaped[0] = -12345.0
    assert dfs.custom_blocks[name][0] == pytest.approx(original)


def test_custom_blocks_empty_when_file_has_none() -> None:
    assert mikeio.read(NO_BLOCKS).custom_blocks == {}
    assert mikeio.Dfs2(NO_BLOCKS).custom_blocks == {}


def test_dataset_custom_blocks_do_not_alias_the_file_object() -> None:
    """Blocks read into a Dataset are copies, so editing them cannot reach the file.

    mikecore returns each block as a view over memory the dfs library frees on
    Close(); writing into a stale view would corrupt freed memory rather than
    raise, so the real assertion is that the file is untouched afterwards.
    """
    dfs = mikeio.Dfs2(ROTATED)
    ds = dfs.read()
    assert ds.custom_blocks[M21_MISC].flags.owndata

    ds.custom_blocks[M21_MISC][3] = 1234.0

    assert dfs.custom_blocks[M21_MISC][3] == pytest.approx(5.0)
    assert dfs.read().custom_blocks[M21_MISC][3] == pytest.approx(5.0)
    assert mikeio.Dfs2(ROTATED).custom_blocks[M21_MISC][3] == pytest.approx(5.0)


def test_reading_custom_blocks_is_lenient() -> None:
    """A foreign writer's odd block must not make the whole file unreadable."""
    from mikeio.dfs._custom_blocks import read_custom_blocks

    class FakeBlock:
        """Stand-in for mikecore.DfsCustomBlock."""

        def __init__(self, name: str, values: np.ndarray) -> None:
            self.Name = name
            self.Values = values

    class FakeFileInfo:
        def __init__(self, blocks: list[FakeBlock]) -> None:
            self.CustomBlocks = blocks

    good = np.array([1.0, 2.0], dtype=np.float32)
    file_info = FakeFileInfo(
        [
            FakeBlock("Good", good),
            FakeBlock("Empty", np.array([], dtype=np.float32)),
            FakeBlock("", np.array([1.0], dtype=np.float32)),
            FakeBlock("Wide", np.zeros((2, 2), dtype=np.float32)),
            FakeBlock("Good", np.array([9.0], dtype=np.float32)),
        ]
    )

    with pytest.warns(UserWarning) as record:
        blocks = read_custom_blocks(file_info)  # type: ignore[arg-type]

    assert list(blocks) == ["Good"]
    np.testing.assert_array_equal(blocks["Good"], good)  # first occurrence wins
    messages = [str(w.message) for w in record]
    assert sum("Duplicate custom block name" in m for m in messages) == 1
    assert sum("Skipping unsupported custom block" in m for m in messages) == 3


# === Writing ===


def test_set_land_value_and_write(tmp_path: Path) -> None:
    """The use case of issue #283: set the MIKE 21 land value of a dfs2."""
    ds = mikeio.read(ROTATED)
    ds.custom_blocks[M21_MISC][3] = -12.5

    fp = tmp_path / "land_value.dfs2"
    ds.to_dfs(fp)

    assert mikeio.Dfs2(fp).custom_blocks[M21_MISC][3] == pytest.approx(-12.5)


def test_add_custom_block_from_a_plain_list(tmp_path: Path) -> None:
    """The documented way to set a land value: a list, written as float32.

    MIKE 21 wants M21_Misc as float32, and a list carries no dtype, so numpy's
    int64/float64 default would be wrong. The Dataset keeps the list as it is and
    the writer reads it as float32.
    """
    expected = [0, 0, -900, -10, 0, 0, 0]

    ds = mikeio.read(NO_BLOCKS)
    ds.custom_blocks = {M21_MISC: expected}
    assert ds.custom_blocks[M21_MISC] == expected  # still a list

    fp = tmp_path / "gebco_from_list.dfs2"
    ds.to_dfs(fp)

    back = mikeio.Dfs2(fp).custom_blocks[M21_MISC]
    assert back.dtype == np.float32
    np.testing.assert_array_equal(back, expected)


def test_write_multiple_custom_blocks_preserves_order(tmp_path: Path) -> None:
    ds = mikeio.read(NO_BLOCKS)
    ds.custom_blocks = {
        "PADDLE PROPERTIES": np.array([0.5, 0.0], dtype=np.float32),
        "MODEL SCALE": np.array([1.0], dtype=np.float32),
        "COUNTS": np.array([1, 2, 3], dtype=np.int32),
    }

    fp = tmp_path / "multi.dfs2"
    ds.to_dfs(fp)

    back = mikeio.Dfs2(fp).custom_blocks
    assert list(back) == ["PADDLE PROPERTIES", "MODEL SCALE", "COUNTS"]
    assert back["COUNTS"].dtype == np.int32
    np.testing.assert_array_equal(back["PADDLE PROPERTIES"], [0.5, 0.0])


@pytest.mark.parametrize(
    "dtype", [np.float32, np.float64, np.int8, np.uint16, np.int32, np.uint32]
)
def test_custom_block_dtype_roundtrip(tmp_path: Path, dtype: type) -> None:
    ds = mikeio.read(NO_BLOCKS)
    values: np.ndarray = np.array([1, 2, 3], dtype=dtype)
    ds.custom_blocks["B"] = values

    fp = tmp_path / f"dtype_{np.dtype(dtype).name}.dfs2"
    ds.to_dfs(fp)

    back = mikeio.Dfs2(fp).custom_blocks["B"]
    assert back.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(back, values)


def test_int16_custom_block_reads_back_as_uint16(tmp_path: Path) -> None:
    """Documents a known mikecore bug: Short is read as c_uint16.

    Writing is correct; the read path picks the wrong ctype, so negative values
    wrap. int16 is accepted rather than rejected because the dfs format supports
    it - the dfs2 user guide tells users to prefer int32.
    """
    ds = mikeio.read(NO_BLOCKS)
    ds.custom_blocks["B"] = np.array([-2, -1, 0, 1], dtype=np.int16)

    fp = tmp_path / "int16.dfs2"
    ds.to_dfs(fp)

    back = mikeio.Dfs2(fp).custom_blocks["B"]
    assert back.dtype == np.uint16
    np.testing.assert_array_equal(back, [65534, 65535, 0, 1])


def test_write_non_contiguous_custom_block(tmp_path: Path) -> None:
    """The dfs C library takes the raw data pointer, ignoring strides."""
    ds = mikeio.read(NO_BLOCKS)
    values = np.arange(10, dtype=np.float32)[::2]

    # assigning into the dict keeps the strides, so the writer has to compact it
    ds.custom_blocks["S"] = values
    assert not ds.custom_blocks["S"].flags.c_contiguous

    fp = tmp_path / "made_contiguous_on_write.dfs2"
    ds.to_dfs(fp)
    np.testing.assert_array_equal(mikeio.Dfs2(fp).custom_blocks["S"], [0, 2, 4, 6, 8])

    # the property setter deep-copies, which compacts it on the way in instead
    ds.custom_blocks = {"S": values}
    assert ds.custom_blocks["S"].flags.c_contiguous


def test_byteswapped_custom_block_is_rejected_on_write(tmp_path: Path) -> None:
    """Byte order is left as given; mikecore rejects the dtype when writing.

    Deliberately not converted: mikecore names the offending dtype, and it does
    so before the file is created, so nothing lands on disk as garbage.
    """
    ds = mikeio.read(NO_BLOCKS)
    ds.custom_blocks = {"S": np.array([1, 2, 3], dtype=">f4")}
    assert not ds.custom_blocks["S"].dtype.isnative

    fp = tmp_path / "byteswapped.dfs2"
    with pytest.raises(Exception, match="not supported"):
        ds.to_dfs(fp)

    assert not fp.exists()


@pytest.mark.parametrize(
    "values,match",
    [
        pytest.param(np.zeros(3, dtype=np.int64), "unsupported dtype", id="dtype"),
        pytest.param(["a", "b"], "could not be read as float32", id="str_list"),
        pytest.param(np.zeros((2, 2), dtype=np.float32), "1-dimensional", id="2d"),
        pytest.param(np.array([], dtype=np.float32), "must not be empty", id="empty"),
    ],
)
def test_invalid_custom_block_values_raise_on_write(
    tmp_path: Path, values: Any, match: str
) -> None:
    """A Dataset takes any value; writing one is where the dfs rules apply."""
    ds = mikeio.read(NO_BLOCKS)
    ds.custom_blocks["B"] = values

    with pytest.raises(ValueError, match=match):
        ds.to_dfs(tmp_path / "invalid.dfs2")


@pytest.mark.parametrize(
    "name,exc,match",
    [
        pytest.param("", ValueError, "must not be empty", id="empty"),
        pytest.param("Ærø", ValueError, "must be ASCII", id="non_ascii"),
        pytest.param("with\x00nul", ValueError, "must not contain NUL", id="nul"),
        pytest.param(1, TypeError, "must be a str", id="not_a_str"),
    ],
)
def test_invalid_custom_block_names_raise_on_write(
    tmp_path: Path, name: Any, exc: type[Exception], match: str
) -> None:
    ds = mikeio.read(NO_BLOCKS)
    ds.custom_blocks[name] = np.array([1.0], dtype=np.float32)

    with pytest.raises(exc, match=match):
        ds.to_dfs(tmp_path / "invalid.dfs2")


# === Other write paths ===


def test_stale_custom_block_is_written_after_geometry_type_changes(
    tmp_path: Path,
) -> None:
    """A Grid2D M21_Misc lands verbatim in the dfs1 that isel(y=0) produces.

    Blocks propagate blindly (see test_dataset.py); whether one still means
    anything after the geometry type changes is the user's call, not MIKE IO's.
    """
    ds = mikeio.read(ROTATED)
    expected = ds.custom_blocks[M21_MISC]

    fp = tmp_path / "stale_block.dfs1"
    ds.isel(y=0).to_dfs(fp)

    np.testing.assert_array_equal(mikeio.Dfs1(fp).custom_blocks[M21_MISC], expected)


def test_read_area_subset_keeps_custom_blocks() -> None:
    """None of the M21_Misc fields depends on the grid extent."""
    dfs = mikeio.Dfs2("tests/testdata/waves.dfs2")
    ds = dfs.read(area=(0.0, 0.0, 500.0, 500.0))

    assert ds.geometry.nx < dfs.geometry.nx
    np.testing.assert_array_equal(
        ds.custom_blocks[M21_MISC], dfs.custom_blocks[M21_MISC]
    )


def test_append_does_not_touch_custom_blocks(tmp_path: Path) -> None:
    ds = mikeio.read(ROTATED)
    fp = tmp_path / "appended.dfs2"
    ds.to_dfs(fp)

    dfs = mikeio.Dfs2(fp)
    dfs.append(ds)

    np.testing.assert_array_equal(
        mikeio.Dfs2(fp).custom_blocks[M21_MISC], ds.custom_blocks[M21_MISC]
    )


def test_dataarray_to_dfs_writes_no_custom_blocks(tmp_path: Path) -> None:
    """DataArray has no dataset-level metadata, so nothing to write."""
    da = mikeio.read(ROTATED)[0]
    fp = tmp_path / "from_dataarray.dfs2"
    da.to_dfs(fp)

    assert mikeio.Dfs2(fp).custom_blocks == {}


def test_dfsu_has_no_custom_blocks(tmp_path: Path) -> None:
    """A dfsu's only block, MIKE_FM, is geometry-derived and stays internal.

    mikecore's DfsuBuilder creates it from the geometry on every write, so it is
    MIKE's to write and not the user's to modify. Exposing it would invite edits
    that are silently discarded, on a block that is wrong after any spatial
    subset anyway. Blocks set on a Dataset are therefore ignored on a dfsu write.
    """
    ds = mikeio.read("tests/testdata/HD2D.dfsu")
    assert ds.custom_blocks == {}

    ds.custom_blocks["Mine"] = np.array([1, 2], dtype=np.int32)
    fp = tmp_path / "with_blocks.dfsu"
    ds.to_dfs(fp)

    assert mikeio.Dfsu2DH(fp).n_items == ds.n_items  # still a valid dfsu
    assert mikeio.read(fp).custom_blocks == {}
