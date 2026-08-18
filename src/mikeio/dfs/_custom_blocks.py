from __future__ import annotations
from typing import Any, Mapping
import warnings

import numpy as np
from numpy.typing import NDArray

from mikecore.DfsBuilder import DfsBuilder
from mikecore.DfsFile import DfsFileInfo


# The dfs header can only store these element types (mikecore.DfsSimpleType). The
# mapping dtype <-> SimpleType is one-to-one, so the array's dtype alone determines
# the on-disk type and back - which is why the SimpleType is never stored separately.
_CUSTOM_BLOCK_DTYPES = (
    "float32",
    "float64",
    "int8",
    "int16",
    "uint16",
    "int32",
    "uint32",
)


def _as_custom_block_array(
    name: str, values: Any, *, copy: bool = False
) -> NDArray[Any]:
    """Validate a custom block and normalize it for the dfs writer.

    mikecore hands the raw data pointer to the dfs C library and ignores both the
    array's strides and its number of dimensions, so anything that is not 1-D and
    C-contiguous is written as silent garbage. Everything that cannot be written
    is therefore rejected here, at assignment time, rather than deep inside
    mikecore when the file is created.

    An ndarray keeps its dtype; anything else is read as float32, which is what
    MIKE 21 expects of the one block most users will ever write.

    Parameters
    ----------
    name:
        Name of the custom block, used in error messages.
    values:
        Values of the custom block. An ndarray of a supported dtype, or any
        sequence that can be read as float32.
    copy:
        Whether the result must own its data. False by default, so that an array
        which needs no conversion is returned as it is.

    Returns
    -------
    numpy.ndarray
        A 1-D, C-contiguous array of native byte order. An array that owns its
        data if *copy*; otherwise *values* itself when it already is one.

    """
    if not isinstance(name, str):
        raise TypeError(f"Custom block name must be a str, not {type(name).__name__}")
    if not name:
        raise ValueError("Custom block name must not be empty")
    if not name.isascii():
        raise ValueError(
            f"Custom block name must be ASCII, got {name!r} "
            "(the dfs library stores block names as ASCII)"
        )
    if "\x00" in name:
        raise ValueError(f"Custom block name must not contain NUL, got {name!r}")

    if isinstance(values, np.ndarray):
        arr = values
    else:
        # A plain sequence carries no dtype, and numpy's default (float64, or int64
        # for an integer literal) is wrong for the block MIKE 21 actually writes:
        # M21_Misc is seven float32 values. Coerce rather than reject, so that
        # ds.custom_blocks["M21_Misc"] = [0, 0, -900, 10, 0, 0, 0] just works.
        try:
            arr = np.asarray(values, dtype=np.float32)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Custom block {name!r} could not be read as float32 values: {e}. "
                "Pass an explicit array for another dtype, "
                "e.g. np.array([...], dtype=np.int32)"
            ) from e

    if arr.dtype.name not in _CUSTOM_BLOCK_DTYPES:
        raise ValueError(
            f"Custom block {name!r} has unsupported dtype '{arr.dtype}'. "
            f"dfs supports: {', '.join(_CUSTOM_BLOCK_DTYPES)}. "
            "Hint: np.array([...], dtype=np.float32)"
        )
    if arr.ndim != 1:
        raise ValueError(
            f"Custom block {name!r} must be 1-dimensional, got shape {arr.shape}"
        )
    if arr.size == 0:
        raise ValueError(f"Custom block {name!r} must not be empty")

    # np.ascontiguousarray alone would keep a non-native byte order ('>f4'), which the
    # dfs library cannot write; the dtype's native twin normalizes that too. Passing
    # arr.dtype itself when it already is native is deliberate: a dtype that is equal
    # but not the identical instance - mikecore hands out such a float32 - makes numpy
    # build a gratuitous view rather than hand back the array as it is.
    dtype = arr.dtype if arr.dtype.isnative else arr.dtype.newbyteorder("=")
    if copy:
        # np.array always copies, so converting and detaching is a single allocation.
        return np.array(arr, dtype=dtype, order="C")
    # asarray rather than array, so that an array which is already fit to be written
    # is stored as it is: an assignment that silently detached the caller's array
    # would be surprising.
    return np.asarray(arr, dtype=dtype, order="C")


def normalize_custom_blocks(blocks: Mapping[str, Any]) -> dict[str, NDArray[Any]]:
    """Validate custom blocks and normalize them for the dfs writer.

    Parameters
    ----------
    blocks:
        Blocks to normalize, name -> values. See *_as_custom_block_array* for the
        accepted values and for what normalizing them means.

    Returns
    -------
    dict[str, numpy.ndarray]
        Block name -> values, each a 1-D, C-contiguous array of native byte
        order. An array that already is one is kept as it is, not copied.

    """
    return {
        name: _as_custom_block_array(name, values) for name, values in blocks.items()
    }


def read_custom_blocks(file_info: DfsFileInfo) -> dict[str, NDArray[Any]]:
    """Read the custom blocks from the header of an *open* dfs file.

    Parameters
    ----------
    file_info:
        FileInfo of a dfs file that has not been closed yet.

    Returns
    -------
    dict[str, numpy.ndarray]
        Block name -> values. The arrays are copies, safe to keep and to modify
        after the file is closed.

    Notes
    -----
    mikecore returns each block's values as a numpy view over memory owned by the
    dfs C library, which is freed by DfsFile.Close(); copy=True below is what makes
    the result safe to keep.

    Reading is deliberately more permissive than writing: a file must not become
    unreadable because it holds a block MIKE IO would refuse to write - a foreign
    writer may well produce one. Anything that cannot be represented (a duplicate
    name, or a block the writer would reject) is skipped with a warning instead of
    raising.

    """
    blocks: dict[str, NDArray[Any]] = {}
    for block in file_info.CustomBlocks:
        if block.Name in blocks:
            warnings.warn(
                f"Duplicate custom block name {block.Name!r} in file; "
                "keeping the first occurrence."
            )
            continue
        try:
            blocks[block.Name] = _as_custom_block_array(
                block.Name, block.Values, copy=True
            )
        except (TypeError, ValueError) as e:
            warnings.warn(f"Skipping unsupported custom block in file: {e}")
    return blocks


def write_custom_blocks(builder: DfsBuilder, blocks: Mapping[str, Any]) -> None:
    """Add custom blocks to the header of a dfs file being created.

    Parameters
    ----------
    builder:
        Builder that has not yet created the file - AddCreateCustomBlock is only
        valid before CreateFile.
    blocks:
        Blocks to write, name -> values, e.g. a Dataset's *custom_blocks*.

    """
    for name, values in blocks.items():
        # The values were normalized when they were assigned, but the mapping and its
        # arrays are mutable, so re-normalize: a non-contiguous or multi-dimensional
        # array is written as silent garbage by the dfs C library.
        values = _as_custom_block_array(name, values)
        # AddCreateCustomBlock derives the dfs SimpleType from the dtype and raises a
        # clear error for an unsupported one; DfsFactory.CreateCustomBlock has no
        # else-branch and fails with UnboundLocalError instead.
        builder.AddCreateCustomBlock(name, values)
