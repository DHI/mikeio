from typing import Optional


class DataDimensionMismatch(ValueError):
    def __init__(self):
        self.message = (
            "Data matrices in the x dimension do not all match in the data list."
            "Data is a list of matrices [t, x]."
        )
        super().__init__(self.message)


class ItemsError(ValueError):
    def __init__(self, n_items_file):
        self.n_items_file = n_items_file
        super().__init__(
            f"'items' must be (a list of) integers between 0 and {n_items_file-1} or str."
        )


class InvalidGeometry(ValueError):
    def __init__(self, message="Invalid operation for this type of geometry"):
        super().__init__(message)


class InvalidDataValueType(ValueError):
    def __init__(self):
        super().__init__(
            "Invalid data type. Choose 'Instantaneous', 'Accumulated', 'StepAccumulated', "
            "'MeanStepBackward', or 'MeanStepForward'"
        )


class OutsideModelDomainError(ValueError):
    def __init__(
        self,
        x: float,
        y: float,
        z: Optional[float] = None,
        message: Optional[str] = None,
    ):
        self.x = x
        self.y = y
        self.z = z
        message = (
            f"Point ({x},{y}) is outside model domain" if message is None else message
        )
        super().__init__(message)
