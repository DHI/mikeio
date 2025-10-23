"""Custom exceptions for mikeio."""

from __future__ import annotations
from typing import Any


class ItemsError(ValueError):
    """Raised when items are not integers or strings."""

    def __init__(self, n_items_file: int) -> None:
        self.n_items_file = n_items_file
        super().__init__(
            f"'items' must be (a list of) integers between 0 and {n_items_file-1} or str."
        )


class OutsideModelDomainError(ValueError):
    """Raised when point(s) are outside the model domain."""

    def __init__(
        self,
        *,
        x: Any,
        y: Any = None,
        z: Any = None,
        indices: Any = None,
        message: str | None = None,
    ) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.indices = indices
        message = (
            f"Point(s) ({x},{y}) with indices: {self.indices} outside model domain"
            if message is None
            else message
        )
        super().__init__(message)
