from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np
from mikecore.DfsuFile import DfsuFileType

from .spatial import (
    GeometryFM3D,
    GeometryFMVerticalProfile,
    Grid2D,
)

if TYPE_CHECKING:  # pragma: no cover
    from .spatial._FM_geometry_layered import _GeometryFMLayered


Mode = Literal["interpolate", "discrete"]


# ---------------------------------------------------------------------------
# Polyline helpers
# ---------------------------------------------------------------------------
def _as_xy(xs: Sequence[float], ys: Sequence[float]) -> np.ndarray:
    ax = np.asarray(xs, dtype=float).ravel()
    ay = np.asarray(ys, dtype=float).ravel()
    if ax.size != ay.size:
        raise ValueError("xs and ys must have the same length")
    if ax.size < 2:
        raise ValueError("the transect needs at least two points")
    return np.column_stack([ax, ay])


def _resample_polyline(pts: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``n`` points equally spaced by arc length along the polyline.

    The arc length is measured in the mesh's own coordinate units (degrees for a
    geographic mesh), matching the MIKE tool (no great-circle correction).
    """
    seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    s = np.linspace(0.0, cum[-1], n)
    x = np.interp(s, cum, pts[:, 0])
    y = np.interp(s, cum, pts[:, 1])
    return np.column_stack([x, y]), s


# ---------------------------------------------------------------------------
# 2D shape-function interpolation
# ---------------------------------------------------------------------------
def _tri_bary_weights(
    corner_xy: np.ndarray, idx: Sequence[int], x: float, y: float
) -> np.ndarray:
    """Barycentric (linear) weights of ``(x, y)`` in the triangle
    ``corner_xy[idx]`` - zeros for the other corners."""
    i1, i2, i3 = idx
    x1, y1 = corner_xy[i1]
    x2, y2 = corner_xy[i2]
    x3, y3 = corner_xy[i3]
    det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    w = np.zeros(len(corner_xy))
    if abs(det) < 1e-20:
        w[i1] = 1.0
        return w
    w1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / det
    w2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / det
    w[i1] = w1
    w[i2] = w2
    w[i3] = 1.0 - w1 - w2
    return w


def _inside_tri(
    corner_xy: np.ndarray, idx: Sequence[int], x: float, y: float, eps: float = 1e-10
) -> bool:
    """Point-in-triangle test."""
    w = _tri_bary_weights(corner_xy, idx, x, y)
    return bool(np.all(w[list(idx)] >= -eps))


def _shape_function_weights(corner_xy: np.ndarray, x: float, y: float) -> np.ndarray:
    """Node weights.

    Triangle: linear barycentric interpolation. Quadrilateral: mean of the two
    diagonal triangulations.
    """
    n = len(corner_xy)
    if n == 3:
        return _tri_bary_weights(corner_xy, (0, 1, 2), x, y)
    if n == 4:
        w1 = (
            _tri_bary_weights(corner_xy, (0, 1, 2), x, y)
            if _inside_tri(corner_xy, (0, 1, 2), x, y)
            else _tri_bary_weights(corner_xy, (0, 2, 3), x, y)
        )
        w2 = (
            _tri_bary_weights(corner_xy, (0, 1, 3), x, y)
            if _inside_tri(corner_xy, (0, 1, 3), x, y)
            else _tri_bary_weights(corner_xy, (1, 2, 3), x, y)
        )
        return 0.5 * (w1 + w2)
    raise ValueError(f"unsupported element with {n} corners")


# ---------------------------------------------------------------------------
# Column geometry
# ---------------------------------------------------------------------------
def _column_level_nodes(geometry: "_GeometryFMLayered", col: int) -> np.ndarray:
    """3D node ids per vertical interface level for a 2D column.

    Shape ``(nl + 1, ncorner)``: level 0 = bed interface, level ``nl`` = surface.
    Corner order matches the source element table.
    """
    e3d = np.asarray(geometry.e2_e3_table[col], dtype=np.int64)
    et = geometry.element_table
    eb = et[e3d[0]]
    half = len(eb) // 2
    levels = np.empty((len(e3d) + 1, half), dtype=np.int64)
    levels[0] = eb[:half]
    for k, e in enumerate(e3d):
        levels[k + 1] = et[e][half:]
    return levels


def _column_corner_xy(geometry: "_GeometryFMLayered", levels: np.ndarray) -> np.ndarray:
    """``(ncorner, 2)`` x,y of a column's corners (from the bed-interface nodes)."""
    return geometry.node_coordinates[levels[0]][:, :2]


# ---------------------------------------------------------------------------
# Transect / 2D-mesh intersection
# ---------------------------------------------------------------------------
def _segment_intersection(
    p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray
) -> np.ndarray | None:
    r = p2 - p1
    s = q2 - q1
    rxs = r[0] * s[1] - r[1] * s[0]
    if abs(rxs) < 1e-14:
        return None
    qp = q1 - p1
    t = (qp[0] * s[1] - qp[1] * s[0]) / rxs
    u = (qp[0] * r[1] - qp[1] * r[0]) / rxs
    if -1e-9 <= t <= 1 + 1e-9 and -1e-9 <= u <= 1 + 1e-9:
        return p1 + t * r
    return None


def _mesh_edges(geometry2d: GeometryFM2D) -> np.ndarray:
    edges = set()
    for el in geometry2d.element_table:
        n = len(el)
        for k in range(n):
            a, b = int(el[k]), int(el[(k + 1) % n])
            edges.add((a, b) if a < b else (b, a))
    return np.array(sorted(edges), dtype=np.int64)


def _find_col(geometry2d: GeometryFM2D, x: float, y: float) -> int:
    try:
        return int(np.atleast_1d(geometry2d.find_index(x=x, y=y))[0])
    except Exception:
        return -1


def _transect_crossings(
    pts: np.ndarray, geometry2d: GeometryFM2D
) -> tuple[np.ndarray, np.ndarray]:
    """Split the transect at every 2D face crossing so each segment lies inside
    one 2D column.

    Returns ``stations`` ``(M + 1, 2)`` and ``seg_col`` ``(M,)`` (2D element id
    per segment, or -1 if outside the domain).
    """
    nodes = geometry2d.node_coordinates[:, :2]
    edges = _mesh_edges(geometry2d)
    ea, eb = nodes[edges[:, 0]], nodes[edges[:, 1]]
    emin = np.minimum(ea, eb)
    emax = np.maximum(ea, eb)
    
    rows: list[tuple[float, float, float]] = []
    base = 0.0
    for i in range(len(pts) - 1):
        p1, p2 = pts[i], pts[i + 1]
        rows.append((base, p1[0], p1[1]))
        # only test edges whose bounding box overlaps the segment's bounding
        # box (a necessary condition for intersection) - avoids an O(n_edges)
        # brute-force scan of the whole mesh per segment
        smin = np.minimum(p1, p2)
        smax = np.maximum(p1, p2)
        cand = np.nonzero(
            (emax[:, 0] >= smin[0])
            & (emin[:, 0] <= smax[0])
            & (emax[:, 1] >= smin[1])
            & (emin[:, 1] <= smax[1])
        )[0]
        for k in cand:
            ip = _segment_intersection(p1, p2, ea[k], eb[k])
            if ip is not None:
                rows.append((base + np.hypot(*(ip - p1)), ip[0], ip[1]))
        base += np.hypot(*(p2 - p1))
    rows.append((base, pts[-1][0], pts[-1][1]))
    
    arr = np.array(rows)
    arr = arr[np.argsort(arr[:, 0])]
    keep = [0]
    for k in range(1, len(arr)):
        if arr[k, 0] - arr[keep[-1], 0] > 1e-9 * max(base, 1.0):
            keep.append(k)
    stations = arr[keep, 1:3]
    
    seg_col = np.array(
        [
            _find_col(geometry2d, *(0.5 * (stations[c] + stations[c + 1])))
            for c in range(len(stations) - 1)
        ],
        dtype=np.int64,
    )
    return stations, seg_col


# ---------------------------------------------------------------------------
# Mode "interpolate": regular distance x elevation grid
# ---------------------------------------------------------------------------
def _extract_interpolated(
    geometry: GeometryFM3D,
    zn: np.ndarray,
    values: list[np.ndarray],
    pts: np.ndarray,
    n_horizontal: int,
    n_vertical: int,
    z_min: float | None,
    z_max: float | None,
) -> tuple[Grid2D, None, list[np.ndarray]]:
    g2 = geometry.geometry2d
    coords, s = _resample_polyline(pts, n_horizontal)
    nh, nv = n_horizontal, n_vertical

    if z_min is None:
        z_min = float(np.min(geometry.node_coordinates[:, 2]))
    if z_max is None:
        z_max = float(np.max(geometry.node_coordinates[:, 2]))
    z_levels = np.linspace(z_min, z_max, nv)

    # --- setup (once): per-column source-node indices, weights, elements ---
    cols: list[tuple | None] = [None] * nh
    for j in range(nh):
        col = _find_col(g2, coords[j, 0], coords[j, 1])
        if col < 0:
            continue
        levels = _column_level_nodes(geometry, col)  # (nl+1, ncorner)
        w = _shape_function_weights(
            _column_corner_xy(geometry, levels), coords[j, 0], coords[j, 1]
        )
        e3d = np.asarray(geometry.e2_e3_table[col], dtype=np.int64)  # (nl,)
        cols[j] = (levels, w, e3d)

    valid = [c for c in cols if c is not None]
    lmax = max((c[2].size for c in valid), default=1)  # max layers/column
    cmax = max((c[1].size for c in valid), default=1)  # max corners

    node_src = np.zeros((nh, lmax + 1, cmax), dtype=np.int64)
    node_w = np.zeros((nh, cmax), dtype=np.float64)
    elem_src = np.zeros((nh, lmax), dtype=np.int64)
    nlev = np.zeros(nh, dtype=np.int64)  # layers per column
    for j, cdat in enumerate(cols):
        if cdat is None:
            continue
        levels, w, e3d = cdat
        nl = e3d.size
        nlev[j] = nl
        node_w[j, : w.size] = w
        node_src[j, : nl + 1, : levels.shape[1]] = levels
        elem_src[j, :nl] = e3d

    # --- per timestep, fully vectorized over the whole time axis ---
    nt = zn.shape[0]
    # interface elevation at every (t, column, level): weighted node z
    iface = np.einsum("tjlc,jc->tjl", zn[:, node_src], node_w)  # (T, nh, lmax+1)
    zc = 0.5 * (iface[:, :, :-1] + iface[:, :, 1:])  # (T, nh, lmax)
    kidx = np.arange(lmax)
    cvalid = kidx[None, :] < nlev[:, None]  # (nh, lmax)
    zc_search = np.where(cvalid[None], zc, np.inf)  # ignore padding
    bed = iface[:, :, 0]  # (T, nh)
    surf = np.take_along_axis(iface, nlev[None, :, None], axis=-1)[:, :, 0]
    lo_cap = np.maximum(nlev - 2, 0)[None, :]
    hi_cap = np.maximum(nlev - 1, 0)[None, :]

    out: list[np.ndarray] = []
    for v in values:
        vals = v[:, elem_src]  # (T, nh, lmax)
        res = np.full((nt, nv, nh), np.nan, dtype=np.float32)
        for m in range(nv):
            z = z_levels[m]
            cnt = np.sum(zc_search <= z, axis=-1)  # (T, nh)
            lo = np.clip(cnt - 1, 0, lo_cap)
            hi = np.minimum(lo + 1, hi_cap)
            zc_lo = np.take_along_axis(zc, lo[:, :, None], -1)[:, :, 0]
            zc_hi = np.take_along_axis(zc, hi[:, :, None], -1)[:, :, 0]
            v_lo = np.take_along_axis(vals, lo[:, :, None], -1)[:, :, 0]
            v_hi = np.take_along_axis(vals, hi[:, :, None], -1)[:, :, 0]
            denom = zc_hi - zc_lo
            wgt = np.clip(np.where(denom != 0, (z - zc_lo) / denom, 0.0), 0.0, 1.0)
            val_m = v_lo + wgt * (v_hi - v_lo)
            inside = (z >= bed) & (z <= surf) & (nlev[None, :] > 0)
            res[:, m, :] = np.where(inside, val_m, np.nan).astype(np.float32)
        out.append(res)

    dx = float(s[1] - s[0]) if nh > 1 else 1.0
    dz = float(z_levels[1] - z_levels[0]) if nv > 1 else 1.0
    grid = Grid2D(
        x0=0.0,
        dx=dx,
        nx=nh,
        y0=z_min,
        dy=dz,
        ny=nv,
        projection="NON-UTM",
        is_vertical=True,
    )
    return grid, None, out


# ---------------------------------------------------------------------------
# Mode "discrete": mesh-following vertical profile
# ---------------------------------------------------------------------------
def _extract_discrete(
    geometry: GeometryFM3D,
    zn: np.ndarray,
    values: list[np.ndarray],
    pts: np.ndarray,
    layer_min: int,
    layer_max: int,
) -> tuple[GeometryFMVerticalProfile, np.ndarray, list[np.ndarray]]:
    g2 = geometry.geometry2d
    maxlayers = geometry.n_layers
    layer_min = max(1, layer_min)
    layer_max = min(maxlayers, layer_max)
    if layer_min > layer_max:
        raise ValueError("layer_min must be <= layer_max")

    stations, seg_col = _transect_crossings(pts, g2)
    n_station = len(stations)
    n_seg = len(seg_col)
    n_lpc = geometry.n_layers_per_column

    # per-station column used to interpolate node z (right segment, else left)
    st_info: dict[int, tuple] = {}
    for si in range(n_station):
        c = seg_col[si] if si < n_seg else -1
        if c < 0 and si > 0:
            c = seg_col[si - 1]
        if c < 0:
            continue
        levels = _column_level_nodes(geometry, int(c))
        corner_xy = _column_corner_xy(geometry, levels)
        w = _shape_function_weights(corner_xy, stations[si, 0], stations[si, 1])
        st_info[si] = (levels, w)

    # build node list + element table (global, surface-aligned layer numbering)
    node_id: dict[tuple[int, int], int] = {}
    node_keys: list[tuple[int, int]] = []
    node_xy: list[tuple[float, float]] = []

    def _node(si: int, gj: int) -> int:
        key = (si, gj)
        if key not in node_id:
            node_id[key] = len(node_xy)
            node_keys.append(key)
            node_xy.append((stations[si, 0], stations[si, 1]))
        return node_id[key]

    element_table: list[np.ndarray] = []
    elem_source: list[int] = []
    for c in range(n_seg):
        col = seg_col[c]
        if col < 0:
            continue
        nl_local = int(n_lpc[col])
        offset = maxlayers - nl_local
        e3d = np.asarray(geometry.e2_e3_table[col], dtype=np.int64)
        for j in range(layer_min, layer_max + 1):
            li = j - offset
            if li < 1 or li > nl_local:
                continue
            bl, br = _node(c, j - 1), _node(c + 1, j - 1)
            tr, tl = _node(c + 1, j), _node(c, j)
            element_table.append(np.array([bl, br, tr, tl], dtype=np.int64))
            elem_source.append(int(e3d[li - 1]))

    if not element_table:
        raise ValueError("the transect does not intersect any mesh column")

    # reorder nodes along the transect (station, then level) for a monotonic
    # relative distance (needed by the vertical-profile plotter)
    order = sorted(range(len(node_keys)), key=lambda i: node_keys[i])
    old_to_new = np.empty(len(order), dtype=np.int64)
    for new_i, old_i in enumerate(order):
        old_to_new[old_i] = new_i
    node_keys = [node_keys[i] for i in order]
    node_xy = [node_xy[i] for i in order]
    node_id = {k: int(old_to_new[v]) for k, v in node_id.items()}
    element_table = [old_to_new[e] for e in element_table]
    elem_source_arr = np.asarray(elem_source, dtype=np.int64)

    n_nodes_out = len(node_xy)

    # --- build padded (node -> source-node-ids, weights) arrays once ---
    cmax = max((len(w) for _, w in st_info.values()), default=1)
    node_src = np.zeros((n_nodes_out, cmax), dtype=np.int64)
    node_w = np.zeros((n_nodes_out, cmax), dtype=np.float64)
    for (si, gj), nid in node_id.items():
        info = st_info.get(si)
        if info is None:
            continue
        levels, w = info
        nl_local = levels.shape[0] - 1
        li = gj - (maxlayers - nl_local)
        if 0 <= li <= nl_local:
            nn = levels[li]
            node_src[nid, : nn.size] = nn
            node_w[nid, : w.size] = w

    # dynamic node z for all timesteps in one einsum (masked nodes -> 0)
    zn_out = np.einsum("tnc,nc->tn", zn[:, node_src], node_w).astype(np.float32)

    # static node z + geometry
    node_coords = np.column_stack([np.array(node_xy), zn_out[0].astype(float)])
    n_sigma_global = geometry.n_sigma_layers
    lowest_sigma = maxlayers - n_sigma_global
    out_layers = np.arange(layer_min, layer_max + 1)
    n_sigma_out = int(np.sum(out_layers > lowest_sigma))
    n_layers_out = layer_max - layer_min + 1
    dfsu_type = (
        DfsuFileType.DfsuVerticalProfileSigma
        if n_sigma_out == n_layers_out
        else DfsuFileType.DfsuVerticalProfileSigmaZ
    )
    geom_out = GeometryFMVerticalProfile(
        node_coordinates=node_coords,
        element_table=[e for e in element_table],
        codes=np.zeros(n_nodes_out, dtype=np.int32),
        projection=geometry.projection_string,
        dfsu_type=dfsu_type,
        n_layers=n_layers_out,
        n_sigma=n_sigma_out,
    )

    # values: a single vectorized gather of the source elements over all time
    out = [v[:, elem_source_arr].astype(np.float32) for v in values]

    return geom_out, zn_out, out


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
def _extract_vertical(
    *,
    geometry: GeometryFM3D,
    zn: np.ndarray | None,
    values: list[np.ndarray],
    mode: Mode,
    xs: Sequence[float],
    ys: Sequence[float],
    layer_min: int | None,
    layer_max: int | None,
    n_horizontal: int,
    n_vertical: int,
    z_min: float | None,
    z_max: float | None,
) -> tuple[GeometryFMVerticalProfile | Grid2D, np.ndarray | None, list[np.ndarray]]:
    """Core vertical-transect extraction (shared by DataArray and Dataset).

    Returns ``(out_geometry, out_zn, out_values)`` where ``out_zn`` is the
    dynamic node z for the ``discrete`` mode and ``None`` for ``interpolate``.
    """
    if not (isinstance(geometry, GeometryFM3D) and geometry.is_layered):
        raise NotImplementedError("extract_vertical requires 3D layered dfsu data")
    if zn is None:
        raise ValueError(
            "the source data has no z-coordinate (zn); a 3D dfsu is required"
        )

    # normalise to a leading time axis (a single-timestep DataArray stores 1-D)
    zn = np.asarray(zn)
    if zn.ndim == 1:
        zn = zn[None, :]
    values = [
        np.asarray(v) if np.asarray(v).ndim == 2 else np.asarray(v)[None, :]
        for v in values
    ]

    pts = _as_xy(xs, ys)

    if mode == "interpolate":
        return _extract_interpolated(
            geometry, zn, values, pts, n_horizontal, n_vertical, z_min, z_max
        )
    if mode == "discrete":
        if layer_min is None or layer_max is None:
            raise ValueError('layer_min and layer_max are required for mode="discrete"')
        return _extract_discrete(geometry, zn, values, pts, layer_min, layer_max)
    raise ValueError(f'mode must be "interpolate" or "discrete", got {mode!r}')
