class DataDimensionMismatch(ValueError):
    def __init__(self):
        self.message = (
            "Data matrices in the x dimension do not all match in the data list."
            "Data is a list of matrices [t, x]."
        )
        super().__init__(self.message)


class ItemNumbersError(ValueError):
    def __init__(self):
        super().__init__(
            "'item_numbers' must be a list or array of values between 0 and 1e15."
        )


class InvalidDataType(ValueError):
    def __init__(self):
        super().__init__("Invalid data type. Choose np.float32 or np.float64")


class InvalidGeometry(ValueError):
    def __init__(self, message="Invalid operation for this type of geometry"):
        super().__init__(message)


class InvalidDataValueType(ValueError):
    def __init__(self):
        super().__init__(
            "Invalid data type. Choose 'Instantaneous', 'Accumulated', 'StepAccumulated', "
            "'MeanStepBackward', or 'MeanStepForward'"
        )
