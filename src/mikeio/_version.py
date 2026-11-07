from __future__ import annotations
from importlib.metadata import PackageNotFoundError, version

try:
    # read version from installed package
    __version__ = version("mikeio")
except PackageNotFoundError:
    # package is not installed
    __version__ = "dev"

__dfs_version__: int = 220
