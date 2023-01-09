import os

from mikecore.DfsuFile import DfsuFile, DfsuFileType

from .dfsu import Dfsu2DH
from .layered import Dfsu2DV, Dfsu3D
from .spectral import DfsuSpectral


class Dfsu:
    def __new__(self, filename, *args, **kwargs):
        filename = str(filename)
        type, dfs = self._get_DfsuFileType_n_Obj(filename)

        if self._type_is_spectral(type):
            return DfsuSpectral(filename, dfs, *args, **kwargs)
        elif self._type_is_2d_horizontal(type):
            return Dfsu2DH(filename, dfs, *args, **kwargs)
        elif self._type_is_2d_vertical(type):
            return Dfsu2DV(filename, dfs, *args, **kwargs)
        elif self._type_is_3d(type):
            return Dfsu3D(filename, dfs, *args, **kwargs)
        else:
            raise ValueError(f"Type {type} is unsupported!")

    @staticmethod
    def _get_DfsuFileType_n_Obj(filename: str):
        ext = os.path.splitext(filename)[-1].lower()
        if "dfs" in ext:
            dfs = DfsuFile.Open(filename)
            type = DfsuFileType(dfs.DfsuFileType)
            # dfs.Close()
        elif "mesh" in ext:
            type = None
            dfs = None
        else:
            raise ValueError(f"{ext} is an unsupported extension")
        return type, dfs

    @staticmethod
    def _type_is_2d_horizontal(type):
        return type in (
            DfsuFileType.Dfsu2D,
            DfsuFileType.DfsuSpectral2D,
            None,
        )

    @staticmethod
    def _type_is_2d_vertical(type):
        return type in (
            DfsuFileType.DfsuVerticalProfileSigma,
            DfsuFileType.DfsuVerticalProfileSigmaZ,
        )

    @staticmethod
    def _type_is_3d(type):
        return type in (
            DfsuFileType.Dfsu3DSigma,
            DfsuFileType.Dfsu3DSigmaZ,
        )

    @staticmethod
    def _type_is_spectral(type):
        """Type is spectral dfsu (point, line or area spectrum)"""
        return type in (
            DfsuFileType.DfsuSpectral0D,
            DfsuFileType.DfsuSpectral1D,
            DfsuFileType.DfsuSpectral2D,
        )
