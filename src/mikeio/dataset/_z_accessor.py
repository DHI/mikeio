"""Z-coordinate accessor for layered dfsu DataArrays."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ._dataarray import DataArray


class ZAccessor:
    """Public read-only access to z-coordinates of a layered DataArray.

    Exposed as ``da.z`` on DataArrays whose geometry is layered (``GeometryFM3D``,
    ``GeometryFMVerticalProfile``, ``GeometryFMVerticalColumn``).

    Parameters
    ----------
    da:
        The parent DataArray whose ``_zn`` array supplies the node z-coordinates.

    Examples
    --------
    ```python
    import mikeio
    ds = mikeio.read("oresund_sigma_z.dfsu")
    da = ds[0]
    zn = da.z.nodes      # (n_time, n_nodes)
    ze = da.z.elements   # (n_time, n_elements) — mean of each element's node z's
    ```

    """

    def __init__(self, da: DataArray) -> None:
        self._da = da
        self._elements_cache: np.ndarray | None = None

    @property
    def nodes(self) -> np.ndarray:
        """Per-timestep z-coordinates at mesh nodes.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_time, n_nodes)`` for time-varying DataArrays, or
            ``(n_nodes,)`` for single-timestep / time-reduced slices.

        """
        zn = self._da._zn
        assert zn is not None  # guaranteed by dispatcher on layered geometries
        return zn

    @property
    def elements(self) -> np.ndarray:
        """Per-timestep z-coordinates at element centers.

        Computed as the mean of each element's node z-coordinates. Cached on
        first access; subsequent accesses return the same array object.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_time, n_elements)`` for time-varying DataArrays, or
            ``(n_elements,)`` for single-timestep / time-reduced slices.

        """
        if self._elements_cache is None:
            self._elements_cache = self._compute_elements()
        return self._elements_cache

    def _compute_elements(self) -> np.ndarray:
        zn = self.nodes
        element_table = self._da.geometry.element_table
        n_elements = len(element_table)

        if zn.ndim == 2:
            n_time = zn.shape[0]
            ze = np.empty((n_time, n_elements), dtype=zn.dtype)
            for j, nodes in enumerate(element_table):
                ze[:, j] = zn[:, np.asarray(nodes, dtype=int)].mean(axis=1)
        else:
            ze = np.empty(n_elements, dtype=zn.dtype)
            for j, nodes in enumerate(element_table):
                ze[j] = zn[np.asarray(nodes, dtype=int)].mean()
        return ze


class NullZAccessor:
    """Placeholder accessor for DataArrays whose geometry has no z-coordinates.

    Any attribute access raises ``AttributeError`` naming the concrete geometry
    type so that users discover the geometry-specific constraint immediately.

    Parameters
    ----------
    da:
        The parent DataArray.

    """

    def __init__(self, da: DataArray) -> None:
        # store geometry class name eagerly so the error message is available
        # even after the parent DataArray is gone
        self._geometry_type_name: str = type(da.geometry).__name__

    def __getattr__(self, name: str) -> Any:
        # __getattr__ is only invoked when normal attribute lookup fails.
        # Dunder probes (deepcopy, IPython repr, etc.) get a plain
        # AttributeError so Python's machinery treats them as a clean miss
        # without spamming tracebacks with the user-facing message.
        if name.startswith("__"):
            raise AttributeError(name)
        # Look up _geometry_type_name via __dict__ directly so that this method
        # cannot recurse when the instance has been constructed without
        # __init__ (e.g. during deepcopy / unpickling).
        geom_name = self.__dict__.get("_geometry_type_name", "<unknown>")
        raise AttributeError(
            f"DataArray with geometry '{geom_name}' has no z-coordinates; "
            "only layered 3D dfsu DataArrays expose .z"
        )
