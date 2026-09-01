"""Global options controlling how MIKE IO objects are displayed."""

from __future__ import annotations

from collections.abc import Sequence
from types import TracebackType
from typing import Any

from .eum import ItemInfo

DISPLAY_MAX_ITEMS = "display_max_items"

OPTIONS: dict[str, Any] = {
    DISPLAY_MAX_ITEMS: 10,
}


def _validate_display_max_items(value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(
            f"{DISPLAY_MAX_ITEMS} must be a non-negative int or None, got {value!r}"
        )


_VALIDATORS = {
    DISPLAY_MAX_ITEMS: _validate_display_max_items,
}


def get_options() -> dict[str, Any]:
    """Get the current global options.

    Returns
    -------
    dict
        Copy of the current options.

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
    display_max_items: int or None
        Maximum number of items listed when printing a Dataset or a dfs file,
        by default 10. Remaining items are summarized on a single line.
        Use None to list all items.

    Examples
    --------
    >>> import mikeio
    >>> mikeio.set_options(display_max_items=100)  # for the rest of the session
    >>> ds = mikeio.read("sw_points.dfs0")  # doctest: +SKIP

    >>> with mikeio.set_options(display_max_items=None):  # doctest: +SKIP
    ...     print(ds)

    See Also
    --------
    get_options

    """

    def __init__(self, **kwargs: Any) -> None:
        self._old: dict[str, Any] = {}
        for key, value in kwargs.items():
            if key not in OPTIONS:
                raise ValueError(
                    f"{key!r} is not a valid option, valid options are: {list(OPTIONS)}"
                )
            _VALIDATORS[key](value)
            self._old[key] = OPTIONS[key]
        OPTIONS.update(kwargs)

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
