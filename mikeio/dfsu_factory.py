import os
from mikecore.DfsuFile import DfsuFile, DfsuFileType
from .dfsu import Dfsu2DH, DfsuSpectral

# from .dfsu_spectral import DfsuSpectral
from .dfsu_layered import Dfsu3D, Dfsu2DV


class Dfsu:
    def __new__(self, filename, *args, **kwargs):
        filename = str(filename)
        type = self._get_DfsuFileType(filename)

        if self._type_is_spectral(type):
            return DfsuSpectral(filename, *args, **kwargs)
        elif self._type_is_2d_horizontal(type):
            return Dfsu2DH(filename, *args, **kwargs)
        elif self._type_is_2d_vertical(type):
            return Dfsu2DV(filename, *args, **kwargs)
        elif self._type_is_3d(type):
            return Dfsu3D(filename, *args, **kwargs)
        else:
            raise ValueError(f"Type {type} is unsupported!")

    @staticmethod
    def _get_DfsuFileType(filename: str):
        ext = os.path.splitext(filename)[-1]
        if "dfs" in ext:
            dfs = DfsuFile.Open(filename)
            type = DfsuFileType(dfs.DfsuFileType)
            dfs.Close()
        elif "mesh" in ext:
            type = None
        else:
            raise ValueError(f"{ext} is an unsupported extension")
        return type

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
