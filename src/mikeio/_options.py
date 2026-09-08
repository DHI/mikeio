"""Global options controlling how MIKE IO behaves."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from types import TracebackType
from typing import Any

from .eum import ItemInfo

DISPLAY_MAX_ITEMS = "display_max_items"
SHOW_PROGRESS = "show_progress"

OPTIONS: dict[str, Any] = {
    DISPLAY_MAX_ITEMS: 10,
    SHOW_PROGRESS: False,
}


def _validate_display_max_items(value: int | None) -> None:
    if value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(
            f"{DISPLAY_MAX_ITEMS} must be a non-negative int or None, got {value!r}"
        )


def _validate_show_progress(value: bool) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{SHOW_PROGRESS} must be a bool, got {value!r}")


class _Unchanged:
    """Marker for "argument not given"; None is a meaningful value."""

    def __repr__(self) -> str:
        return "<unchanged>"


# Typed Any so the signature shows the real type: int | None
_unchanged: Any = _Unchanged()


def get_options() -> dict[str, Any]:
    """Get the current global options.

    Returns
    -------
    dict
        Copy of the current options, e.g. to restore them later with
        `set_options(**old)`.

    See Also
    --------
    set_options

    """
    return dict(OPTIONS)


class set_options:
    """Set global options for MIKE IO.

    Can be used to change options for the rest of the session, or as a
    context manager to change them temporarily.

    Parameters
    ----------
    display_max_items: int or None, optional
        Maximum number of items listed when printing a Dataset or a dfs file,
        by default 10. Remaining items are summarized on a single line.
        Use None to list all items, or 0 to print only the number of items.
    show_progress: bool, optional
        Show a progress bar while reading or writing many timesteps,
        by default False.

    Examples
    --------
    >>> import mikeio
    >>> mikeio.set_options(display_max_items=100)  # for the rest of the session
    >>> ds = mikeio.read("sw_points.dfs0")
    >>> with mikeio.set_options(display_max_items=None):
    ...     print(ds)
    >>> with mikeio.set_options(show_progress=True):
    ...     ds = mikeio.read("big.dfsu")

    See Also
    --------
    get_options

    """

    def __init__(
        self,
        *,
        display_max_items: int | None = _unchanged,
        show_progress: bool = _unchanged,
    ) -> None:
        self._old: dict[str, Any] = {}
        self._set(DISPLAY_MAX_ITEMS, display_max_items, _validate_display_max_items)
        self._set(SHOW_PROGRESS, show_progress, _validate_show_progress)

    def _set(self, key: str, value: Any, validate: Callable[[Any], None]) -> None:
        if value is _unchanged:
            return
        validate(value)
        self._old[key] = OPTIONS[key]
        OPTIONS[key] = value

    def __enter__(self) -> set_options:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        OPTIONS.update(self._old)


def _item_txt(items: Sequence[ItemInfo]) -> list[str]:
    """Format items for display, truncated according to display_max_items."""
    n_items = len(items)
    max_items = OPTIONS[DISPLAY_MAX_ITEMS]
    n_shown = n_items if max_items is None else min(n_items, max_items)

    if n_shown == 0:
        return [f"number of items: {n_items}"]

    out = ["items:"]
    out.extend(f"  {i}:  {item}" for i, item in enumerate(items[:n_shown]))
    if n_shown < n_items:
        out.append(f"  ... and {n_items - n_shown} more items ({n_items} total)")
    return out


def _show_progress() -> bool:
    """Whether to show a progress bar for long-running operations."""
    return bool(OPTIONS[SHOW_PROGRESS])
