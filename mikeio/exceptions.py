from __future__ import annotations
from typing import Any


class DataDimensionMismatch(ValueError):
    def __init__(self) -> None:
        self.message = (
            "Data matrices in the x dimension do not all match in the data list."
            "Data is a list of matrices [t, x]."
        )
        super().__init__(self.message)


class ItemsError(ValueError):
    def __init__(self, n_items_file: int) -> None:
        self.n_items_file = n_items_file
        super().__init__(
            f"'items' must be (a list of) integers between 0 and {n_items_file-1} or str."
        )


class InvalidGeometry(ValueError):
    def __init__(
        self, message: str = "Invalid operation for this type of geometry"
    ) -> None:
        super().__init__(message)


class InvalidDataValueType(ValueError):
    def __init__(self) -> None:
        super().__init__(
            "Invalid data type. Choose 'Instantaneous', 'Accumulated', 'StepAccumulated', "
            "'MeanStepBackward', or 'MeanStepForward'"
        )


class OutsideModelDomainError(ValueError):
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
