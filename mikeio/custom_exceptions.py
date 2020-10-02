class FileDoesNotExist(Exception):
    def __init__(self, filename=""):
        super().__init__(f"File {filename} does not exist.")


class DataDimensionMismatch(ValueError):
    def __init__(self):
        self.message = "Data matrices in the x dimension do not all match in the data list." \
                       "Data is a list of matrices [t, x]."
        super().__init__(self.message)
