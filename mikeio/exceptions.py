class DataDimensionMismatch(ValueError):
    def __init__(self):
        self.message = (
            "Data matrices in the x dimension do not all match in the data list."
            "Data is a list of matrices [t, x]."
        )
        super().__init__(self.message)


class ItemNumbersError(ValueError):
    def __init__(self, n_items_file):
        super().__init__(
            f"item numbers must be (a list of) integers between 0 and {n_items_file-1}."
        )


class ItemsError(ValueError):
    def __init__(self, n_items_file):
        super().__init__(
            f"'items' must be (a list of) integers between 0 and {n_items_file-1} or str."
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


class NoDataForQuery(ValueError):
    def __init__(self, query_string):
        super().__init__(f"Invalid query {query_string}")


class InvalidQuantity(ValueError):
    def __init__(self, message="Invalid quantity."):
        super().__init__(message)
