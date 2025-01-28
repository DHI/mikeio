from __future__ import annotations
from pathlib import Path
from typing import Any
from mikecore.DfsuFile import DfsuFile, DfsuFileType

from ._dfsu import Dfsu2DH
from ._layered import Dfsu2DV, Dfsu3D
from ._spectral import DfsuSpectral


DFSU_MAPPING = {
    # DfsuFileType.Dfsu1D: None,
    DfsuFileType.Dfsu2D: Dfsu2DH,
    # DfsuFileType.DfsuVerticalColumn: None,
    DfsuFileType.DfsuVerticalProfileSigma: Dfsu2DV,
    DfsuFileType.DfsuVerticalProfileSigmaZ: Dfsu2DV,
    DfsuFileType.Dfsu3DSigma: Dfsu3D,
    DfsuFileType.Dfsu3DSigmaZ: Dfsu3D,
    DfsuFileType.DfsuSpectral0D: DfsuSpectral,
    DfsuFileType.DfsuSpectral1D: DfsuSpectral,
    DfsuFileType.DfsuSpectral2D: DfsuSpectral,
}


def dfsu(filename: str | Path) -> Any:
    filename = str(filename)
    dfs = DfsuFile.Open(filename)
    type = DfsuFileType(dfs.DfsuFileType)
    dfs.Close()

    klass = DFSU_MAPPING.get(type)

    if klass is None:
        raise ValueError(f"Unsupported dfsu type: {type}")

    return klass(filename)


class Dfsu:
    """Factory class for dfsu files."""

    def __new__(self, filename: str | Path) -> Any:
        return dfsu(filename)
