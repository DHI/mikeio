from ._dfsu import Mesh, _write_dfsu
from ._factory import Dfsu

# Alias for fmskill/modelskill
from ._dfsu import Dfsu2DH

_Dfsu = Dfsu2DH

__all__ = ["Mesh", "Dfsu", "_write_dfsu", "Dfsu2DH", "_Dfsu"]
